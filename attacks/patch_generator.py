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
from .patch_losses import gt_matched_topk_suppression_loss, target_suppression_loss
from .patch_transforms import transform_patch


@dataclass
class PatchResult:
    patch: Tensor
    losses: list[float]
    before_score: float
    after_score: float
    gradient_norms: list[float]


class PatchGenerator:
    """Optimize only a patch tensor against a frozen detector."""

    def __init__(self, detector: torch.nn.Module, config: PatchConfig | None = None, device: torch.device | None = None) -> None:
        self.detector = detector.eval()
        self.config = config or PatchConfig()
        self.device = device or next(detector.parameters()).device
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        self._generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

    def _place(self, patch: Tensor, image: Tensor, box: Tensor) -> Tensor:
        _, _, height, width = image.shape
        _, _, patch_height, patch_width = patch.shape
        _, _, box_width, box_height = box * width
        target_width = max(8, int(float(box_width * self.config.patch_relative_scale)))
        target_height = max(8, int(float(box_height * self.config.patch_relative_scale)))
        resized = F.interpolate(patch, size=(min(height, target_height), min(width, target_width)), mode="bilinear", align_corners=False)
        left = int(float(box[0] * width - resized.shape[-1] / 2)).__index__()
        center_y = box[1] * height
        if self.config.placement_mode == "torso":
            center_y = (box[1] - box[3] / 2 + box[3] * self.config.torso_relative_y) * height
        top = int(float(center_y - resized.shape[-2] / 2)).__index__()
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

    def _score(self, detector: torch.nn.Module, image: Tensor, boxes: Tensor, classes: Tensor, image_size: int) -> Tensor:
        raw = detector(image)
        raw_prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
        if self.config.objective == "v3":
            return gt_matched_topk_suppression_loss(raw_prediction, boxes, classes, image_size, self.config.candidate_iou_threshold, self.config.candidate_top_k)
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
        gradient_norms: list[float] = []
        for _ in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            patched = self._patched(image, boxes, patch.clamp(0, 1))
            loss = self._score(self.detector, patched, boxes, classes, image.shape[-1])
            loss.backward()
            gradient_norms.append(float(patch.grad.norm().detach()))
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
            losses.append(float(loss.detach()))
        with torch.no_grad():
            after = float(self._score(self.detector, self._patched(image, boxes, patch), boxes, classes, image.shape[-1]))
        return PatchResult(patch.detach(), losses, before, after, gradient_norms)

    def generate_many(self, images: list[Tensor], boxes: list[Tensor], classes: list[Tensor], batch_size: int = 8) -> PatchResult:
        """Optimize one shared patch over a reproducible source-image set."""
        if not images or len(images) != len(boxes) or len(images) != len(classes):
            raise ValueError("images, boxes, and classes must be non-empty and aligned")
        patch = torch.rand((1, 3, self.config.patch_size, self.config.patch_size), device=self.device, generator=self._generator, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.config.learning_rate)
        with torch.no_grad():
            before = sum(float(self._score(self.detector, image.to(self.device), box.to(self.device), cls.to(self.device), image.shape[-1])) for image, box, cls in zip(images, boxes, classes)) / len(images)
        losses: list[float] = []
        gradient_norms: list[float] = []
        for _ in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            step_value = 0.0
            start = (_ * batch_size) % len(images)
            indices = [(start + offset) % len(images) for offset in range(min(batch_size, len(images)))]
            batch_losses = []
            for index in indices:
                image, box, cls = images[index].to(self.device), boxes[index].to(self.device), classes[index].to(self.device)
                patched = self._patched(image, box, patch.clamp(0, 1))
                batch_losses.append(self._score(self.detector, patched, box, cls, image.shape[-1]))
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            gradient_norms.append(float(patch.grad.norm().detach()))
            step_value = float(loss.detach())
            optimizer.step()
            with torch.no_grad(): patch.clamp_(0, 1)
            losses.append(step_value)
        with torch.no_grad():
            after = sum(float(self._score(self.detector, self._patched(image.to(self.device), box.to(self.device), patch), box.to(self.device), cls.to(self.device), image.shape[-1])) for image, box, cls in zip(images, boxes, classes)) / len(images)
        return PatchResult(patch.detach(), losses, before, after, gradient_norms)

    @staticmethod
    def save(result: PatchResult, output: Path, metadata: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        pixels = (result.patch[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")
        Image.fromarray(pixels).save(output)
        torch.save(result.patch.cpu(), output.with_suffix(".pt"))
        payload = dict(metadata)
        payload.update({"before_score": result.before_score, "after_score": result.after_score, "loss_last": result.losses[-1] if result.losses else None, "gradient_norm_last": result.gradient_norms[-1] if result.gradient_norms else None})
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
