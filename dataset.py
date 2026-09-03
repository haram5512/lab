"""COCO-person loading with on-the-fly adversarial patch composition.

The dataset keeps only source patch files on disk.  A transformed patch and
its pixel mask are created in ``__getitem__`` so the same image can produce a
different view on every epoch.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


PatchMode = Literal["clean", "seen", "unseen"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CocoPersonConfig:
    images_dir: Path
    annotations_file: Path
    image_size: int = 640
    patch_mode: PatchMode = "clean"
    patch_dirs: tuple[Path, ...] = ()
    patch_probability: float = 0.5
    max_images: int | None = None
    seed: int = 7


class CocoPersonPatchDataset(Dataset[dict[str, Any]]):
    """COCO detection samples restricted to category id 1 (person).

    ``patch_mode`` is intentionally explicit:

    - ``clean``: never compose a patch
    - ``seen``: compose only training patch directories
    - ``unseen``: compose only held-out evaluation patch directories

    Bounding boxes remain the original person boxes after patch composition;
    this models an occluding adversarial patch while retaining the detection
    target.  Coordinates returned to the model are normalized ``xywh``.
    """

    def __init__(self, config: CocoPersonConfig) -> None:
        if not 0.0 <= config.patch_probability <= 1.0:
            raise ValueError("patch_probability must be in [0, 1].")
        if config.patch_mode != "clean" and not config.patch_dirs:
            raise ValueError(f"patch_mode={config.patch_mode!r} requires patch_dirs.")

        self.config = config
        payload = json.loads(Path(config.annotations_file).read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in payload["images"]}
        annotations: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
        for annotation in payload.get("annotations", []):
            if int(annotation.get("category_id", -1)) == 1:
                bbox = annotation.get("bbox", [])
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                    annotations[int(annotation["image_id"])].append(annotation)

        self.records = [
            (images[image_id], annotations[image_id])
            for image_id in sorted(images)
            if annotations[image_id]
        ]
        if config.max_images is not None:
            self.records = self.records[: config.max_images]
        if not self.records:
            raise ValueError("No COCO images with person annotations were found.")

        self.patch_files = self._find_patch_files(config.patch_dirs)
        if config.patch_mode != "clean" and not self.patch_files:
            raise ValueError("No image files found in the selected patch directories.")

    @staticmethod
    def _find_patch_files(directories: Sequence[Path]) -> list[Path]:
        return sorted(
            path
            for directory in directories
            for path in Path(directory).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.config.seed + index * 1_000_003)

    def _should_patch(self, rng: random.Random) -> bool:
        return self.config.patch_mode != "clean" and rng.random() < self.config.patch_probability

    def _compose_patch(
        self, image: Image.Image, rng: random.Random
    ) -> tuple[Image.Image, Image.Image]:
        patch_path = self.patch_files[rng.randrange(len(self.patch_files))]
        patch = Image.open(patch_path).convert("RGBA")
        patch.thumbnail((max(8, image.width // 2), max(8, image.height // 2)), Image.Resampling.LANCZOS)

        scale = rng.uniform(0.35, 1.0)
        patch = patch.resize(
            (max(8, int(patch.width * scale)), max(8, int(patch.height * scale))),
            Image.Resampling.BICUBIC,
        )
        patch = patch.rotate(rng.uniform(-35.0, 35.0), expand=True, resample=Image.Resampling.BICUBIC)
        patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(0.65, 1.35))

        # A mild perspective-like quadrilateral warp using PIL's perspective
        # transform.  The random coefficient is bounded to keep the patch
        # visible and avoid unstable extreme geometries.
        if rng.random() < 0.7 and patch.width > 4 and patch.height > 4:
            w, h = patch.size
            jitter = min(w, h) * rng.uniform(0.02, 0.12)
            quad = [
                (rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)),
                (w + rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)),
                (w + rng.uniform(-jitter, jitter), h + rng.uniform(-jitter, jitter)),
                (rng.uniform(-jitter, jitter), h + rng.uniform(-jitter, jitter)),
            ]
            patch = patch.transform(patch.size, Image.Transform.QUAD, sum(quad, ()), Image.Resampling.BICUBIC)

        max_x = max(0, image.width - patch.width)
        max_y = max(0, image.height - patch.height)
        left = rng.randint(0, max_x)
        top = rng.randint(0, max_y)
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layer.alpha_composite(patch, (left, top))
        composed = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
        mask = Image.new("L", image.size, 0)
        patch_alpha = patch.getchannel("A")
        mask.paste(patch_alpha, (left, top))
        return composed, mask

    def _letterbox(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image, float, int, int]:
        """Resize with aspect-ratio preservation and centered padding."""
        target = self.config.image_size
        original_width, original_height = image.size
        scale = min(target / original_width, target / original_height)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        pad_left = (target - resized_width) // 2
        pad_top = (target - resized_height) // 2
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (target, target), (114, 114, 114))
        canvas.paste(resized, (pad_left, pad_top))
        mask_canvas = Image.new("L", (target, target), 0)
        mask_canvas.paste(resized_mask, (pad_left, pad_top))
        return canvas, mask_canvas, scale, pad_left, pad_top

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, raw_annotations = self.records[index]
        image_path = Path(self.config.images_dir) / record["file_name"]
        image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        original_width, original_height = image.size
        rng = self._rng(index)
        patched = False
        patch_mask = Image.new("L", image.size, 0)
        if self._should_patch(rng):
            image, patch_mask = self._compose_patch(image, rng)
            patched = True

        image, patch_mask, scale, pad_left, pad_top = self._letterbox(image, patch_mask)
        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.asarray(patch_mask).copy()).unsqueeze(0).float() / 255.0

        boxes = []
        for annotation in raw_annotations:
            x, y, width, height = map(float, annotation["bbox"])
            x1 = x * scale + pad_left
            y1 = y * scale + pad_top
            x2 = (x + width) * scale + pad_left
            y2 = (y + height) * scale + pad_top
            boxes.append(
                [
                    ((x1 + x2) / 2) / self.config.image_size,
                    ((y1 + y2) / 2) / self.config.image_size,
                    (x2 - x1) / self.config.image_size,
                    (y2 - y1) / self.config.image_size,
                ]
            )
        return {
            "img": image_tensor,
            "bboxes": torch.tensor(boxes, dtype=torch.float32),
            "cls": torch.zeros((len(boxes), 1), dtype=torch.float32),
            "patch_mask": mask_tensor,
            "patched": patched,
            "image_id": int(record["id"]),
            "file_name": record["file_name"],
            "original_size": torch.tensor([original_width, original_height], dtype=torch.float32),
            "letterbox_scale": torch.tensor(scale, dtype=torch.float32),
            "letterbox_pad": torch.tensor([pad_left, pad_top], dtype=torch.float32),
        }


def coco_person_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length person targets into the model's loss contract."""
    images = torch.stack([item["img"] for item in batch])
    bboxes = torch.cat([item["bboxes"] for item in batch], dim=0)
    classes = torch.cat([item["cls"] for item in batch], dim=0)
    batch_idx = torch.cat(
        [torch.full((len(item["bboxes"]),), index, dtype=torch.long) for index, item in enumerate(batch)]
    )
    return {
        "img": images,
        "bboxes": bboxes,
        "cls": classes,
        "batch_idx": batch_idx,
        "patch_mask": torch.stack([item["patch_mask"] for item in batch]),
        "patched": torch.tensor([item["patched"] for item in batch], dtype=torch.bool),
        "image_id": [item["image_id"] for item in batch],
        "file_name": [item["file_name"] for item in batch],
        "original_size": torch.stack([item["original_size"] for item in batch]),
        "letterbox_scale": torch.stack([item["letterbox_scale"] for item in batch]),
        "letterbox_pad": torch.stack([item["letterbox_pad"] for item in batch]),
    }
