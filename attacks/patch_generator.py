from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from .patch_config import PatchConfig
from .patch_losses import target_suppression_loss
from .patch_transforms import transform_patch


@dataclass
class PatchResult:
    patch: Tensor
    losses: list[float]
    before_score: float
    after_score: float


class PatchGenerator:
    """Optimize only a patch tensor against a frozen detector."""

    def __init__(self, detector: torch.nn.Module, config: PatchConfig | None = None, device: torch.device | None = None) -> None:
        self.detector = detector.eval()
        self.config = config or PatchConfig()
        self.device = device or next(detector.parameters()).device
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        self._generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

    @staticmethod
    def _place(patch: Tensor, image: Tensor, box: Tensor) -> Tensor:
        _, _, height, width = image.shape
        _, _, patch_height, patch_width = patch.shape
        _, _, box_width, box_height = box * width
        target_width = max(8, int(float(box_width * 0.35)))
        target_height = max(8, int(float(box_height * 0.35)))
        resized = F.interpolate(patch, size=(min(height, target_height), min(width, target_width)), mode="bilinear", align_corners=False)
        left = int(float(box[0] * width - resized.shape[-1] / 2)).__index__()
        top = int(float(box[1] * height - resized.shape[-2] / 2)).__index__()
        left = max(0, min(width - resized.shape[-1], left))
        top = max(0, min(height - resized.shape[-2], top))
        canvas = image.clone()
        canvas[:, :, top : top + resized.shape[-2], left : left + resized.shape[-1]] = resized
        return canvas

    def _patched(self, image: Tensor, boxes: Tensor, patch: Tensor) -> Tensor:
        patched = image.clone()
        for index in range(image.shape[0]):
            transformed = transform_patch(patch, self.config.eot, self._generator)
            patched[index : index + 1] = self._place(transformed, patched[index : index + 1], boxes[index, 0])
        return patched

    @staticmethod
    def _score(detector: torch.nn.Module, image: Tensor, boxes: Tensor, classes: Tensor, image_size: int) -> Tensor:
        raw = detector(image)
        raw_prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
        return target_suppression_loss(raw_prediction, boxes, classes, image_size)

    def generate(self, image: Tensor, boxes: Tensor, classes: Tensor) -> PatchResult:
        image = image.to(self.device)
        boxes = boxes.to(self.device)
        classes = classes.to(self.device)
        random.seed(self.config.seed)
        patch = torch.rand((1, 3, self.config.patch_size, self.config.patch_size), device=self.device, generator=self._generator, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.config.learning_rate)
        with torch.no_grad():
            before = float(self._score(self.detector, image, boxes, classes, image.shape[-1]))
        losses: list[float] = []
        for _ in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            patched = self._patched(image, boxes, patch.clamp(0, 1))
            loss = self._score(self.detector, patched, boxes, classes, image.shape[-1])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
            losses.append(float(loss.detach()))
        with torch.no_grad():
            after = float(self._score(self.detector, self._patched(image, boxes, patch), boxes, classes, image.shape[-1]))
        return PatchResult(patch.detach(), losses, before, after)

    def generate_many(self, images: list[Tensor], boxes: list[Tensor], classes: list[Tensor], batch_size: int = 8) -> PatchResult:
        """Optimize one shared patch over a reproducible source-image set."""
        if not images or len(images) != len(boxes) or len(images) != len(classes):
            raise ValueError("images, boxes, and classes must be non-empty and aligned")
        patch = torch.rand((1, 3, self.config.patch_size, self.config.patch_size), device=self.device, generator=self._generator, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.config.learning_rate)
        def batches() -> list[tuple[Tensor, Tensor, Tensor]]:
            return [
                (torch.cat(images[start:start + batch_size]).to(self.device), torch.cat(boxes[start:start + batch_size]).to(self.device), torch.cat(classes[start:start + batch_size]).to(self.device))
                for start in range(0, len(images), batch_size)
            ]
        source_batches = batches()
        with torch.no_grad():
            before = sum(float(self._score(self.detector, image, box, cls, image.shape[-1])) for image, box, cls in source_batches) / len(source_batches)
        losses: list[float] = []
        for _ in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[Tensor] = []
            for image, box, cls in source_batches:
                patched = self._patched(image, box, patch.clamp(0, 1))
                step_losses.append(self._score(self.detector, patched, box, cls, image.shape[-1]) / len(source_batches))
            loss = torch.stack(step_losses).sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad(): patch.clamp_(0, 1)
            losses.append(float(loss.detach()))
        with torch.no_grad():
            after = sum(float(self._score(self.detector, self._patched(image, box, patch), box, cls, image.shape[-1])) for image, box, cls in source_batches) / len(source_batches)
        return PatchResult(patch.detach(), losses, before, after)

    @staticmethod
    def save(result: PatchResult, output: Path, metadata: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        pixels = (result.patch[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")
        Image.fromarray(pixels).save(output)
        payload = dict(metadata)
        payload.update({"before_score": result.before_score, "after_score": result.after_score, "loss_last": result.losses[-1] if result.losses else None})
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
