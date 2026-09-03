"""Plain RGB Ultralytics detector used as the clean-training baseline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression


class BaselineDetector(nn.Module):
    """Run the unmodified Ultralytics RGB graph with differentiable outputs."""

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        super().__init__()
        loaded = YOLO(weights, task="detect").model
        if not isinstance(loaded, DetectionModel):
            raise TypeError(f"{weights!r} is not an Ultralytics detection model.")
        self.detector = loaded
        for parameter in self.detector.parameters():
            parameter.requires_grad_(True)
        existing = getattr(self.detector, "args", {})
        if not isinstance(existing, dict):
            existing = vars(existing)
        self.detector.args = get_cfg(overrides=existing)

    def forward(self, images: Tensor) -> Any:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W].")
        cache: list[Any] = []
        x: Any = images
        for module in self.detector.model:
            if module.f != -1:
                if isinstance(module.f, int):
                    x = cache[module.f]
                else:
                    x = [x if index == -1 else cache[index] for index in module.f]
            x = module(x)
            cache.append(x if module.i in self.detector.save else None)
        return x

    def loss(self, batch: dict[str, Tensor], predictions: Any) -> tuple[Tensor, Any]:
        return self.detector.loss(batch, predictions)

    def forward_for_loss(self, images: Tensor) -> Any:
        """Produce raw detection outputs while keeping the rest in eval mode."""
        detect = self.detector.model[-1]
        was_training = detect.training
        detect.train()
        try:
            return self.forward(images)
        finally:
            detect.train(was_training)

    @torch.no_grad()
    def predict(self, images: Tensor, confidence: float = 0.001, iou: float = 0.7) -> list[Tensor]:
        """Return NMS-filtered ``xyxy, confidence, class`` rows."""
        was_training = self.training
        self.eval()
        raw = self.forward(images)
        prediction = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(
            prediction,
            conf_thres=confidence,
            iou_thres=iou,
            classes=[0],
            max_det=300,
            nc=80,
        )
        if was_training:
            self.train()
        return detections
