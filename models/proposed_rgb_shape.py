"""Lightweight P4-only RGB/shape fusion for YOLO11n."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / "runs" / ".ultralytics"))
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression


@dataclass
class ProposedV1Config:
    weights: str = "yolo11n.pt"
    shape_channels: tuple[int, int, int, int] = (8, 16, 24, 32)
    fusion_alpha: float = 0.05
    fusion_layer: int = 19
    num_classes: int = 80


class ConvBNAct(nn.Sequential):
    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(c1, c2, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class SobelMagnitude(nn.Module):
    """Differentiable, per-image normalized single-channel edge magnitude."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sx.transpose(-1, -2).contiguous())
        self.eps = eps

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        gray = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + self.eps)
        maximum = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return magnitude / maximum


class LightweightShapeP4(nn.Module):
    """Four stride-2 stages producing a compact stride-16 P4 tensor."""

    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stages = nn.Sequential(
            ConvBNAct(1, c1, 2),
            ConvBNAct(c1, c2, 2),
            ConvBNAct(c2, c3, 2),
            ConvBNAct(c3, c4, 2),
        )

    def forward(self, shape: Tensor) -> Tensor:
        return self.stages(shape)


class ResidualP4Fusion(nn.Module):
    """Preserve pretrained RGB P4 while learning a small shape residual."""

    def __init__(self, shape_channels: int, rgb_channels: int, alpha: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(shape_channels, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, rgb: Tensor, shape: Tensor) -> Tensor:
        if rgb.shape[-2:] != shape.shape[-2:]:
            raise ValueError(f"P4 spatial mismatch: RGB={rgb.shape}, shape={shape.shape}")
        return rgb + self.alpha * self.projection(shape)


class ProposedRGBShapeV1(nn.Module):
    """YOLO11n with differentiable Sobel and one lightweight P4 fusion."""

    def __init__(self, config: ProposedV1Config | None = None) -> None:
        super().__init__()
        self.config = config or ProposedV1Config()
        loaded = YOLO(self.config.weights, task="detect").model
        if not isinstance(loaded, DetectionModel):
            raise TypeError("weights must load an Ultralytics DetectionModel")
        for parameter in loaded.parameters():
            parameter.requires_grad_(True)
        if int(loaded.model[-1].nc) != self.config.num_classes:
            raise ValueError("v1 retains the pretrained 80-class detection head")
        existing = getattr(loaded, "args", {})
        loaded.args = get_cfg(overrides=existing if isinstance(existing, dict) else vars(existing))
        self.detector = loaded
        self.detect = self.detector.model[-1]
        if not isinstance(self.detect, Detect) or list(self.detect.f) != [16, 19, 22]:
            raise ValueError("Expected YOLO11n Detect inputs [16,19,22]")
        p4_channels = int(self.detect.cv2[1][0].conv.in_channels)
        self.shape_extractor = SobelMagnitude()
        self.shape_backbone = LightweightShapeP4(self.config.shape_channels)
        self.fusion = ResidualP4Fusion(self.config.shape_channels[-1], p4_channels, self.config.fusion_alpha)
        self.last_diagnostics: dict[str, Any] = {}
        self.initialization_report = {
            "pretrained_parameter_count": sum(p.numel() for p in self.detector.parameters()),
            "new_parameter_count": sum(p.numel() for p in self.shape_backbone.parameters()) + sum(p.numel() for p in self.fusion.parameters()),
            "incompatible_keys": [],
            "config": asdict(self.config),
        }

    def forward(self, rgb: Tensor) -> Any:
        shape_input = self.shape_extractor(rgb)
        shape_p4 = self.shape_backbone(shape_input)
        cache: list[Any] = []
        x: Any = rgb
        rgb_p4 = fused_p4 = None
        for module in self.detector.model:
            if module.f != -1:
                if isinstance(module.f, int):
                    x = cache[module.f]
                else:
                    x = [x if index == -1 else cache[index] for index in module.f]
            x = module(x)
            if module.i == self.config.fusion_layer:
                rgb_p4 = x
                fused_p4 = self.fusion(rgb_p4, shape_p4)
                x = fused_p4
            cache.append(x if module.i in self.detector.save else None)
        self.last_diagnostics = {
            "rgb_input": tuple(rgb.shape),
            "shape_input": tuple(shape_input.shape),
            "rgb_p4": tuple(rgb_p4.shape) if rgb_p4 is not None else None,
            "shape_p4": tuple(shape_p4.shape),
            "fused_p4": tuple(fused_p4.shape) if fused_p4 is not None else None,
            "detection_type": type(x).__name__,
        }
        return x

    def loss(self, batch: dict[str, Tensor], predictions: Any) -> tuple[Tensor, Any]:
        return self.detector.loss(batch, predictions)

    @torch.no_grad()
    def predict(self, images: Tensor, confidence: float = 0.001, iou: float = 0.7,
                classes: list[int] | None = None) -> list[Tensor]:
        was_training = self.training
        self.eval()
        raw = self(images)
        prediction = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(prediction, conf_thres=confidence, iou_thres=iou,
                                         classes=classes, max_det=300, nc=self.config.num_classes)
        self.train(was_training)
        return detections
