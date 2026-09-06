"""Multi-scale Sobel RGB/shape fusion for YOLO11n.

The RGB detector remains pretrained and receives small residual shape features
at P3/P4/P5.  The shape branch is shared and lightweight; each fusion has its
own learnable scalar initialized close to zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
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

from .proposed_rgb_shape import SobelMagnitude, ConvBNAct


@dataclass
class ProposedV2Config:
    weights: str = "yolo11n.pt"
    shape_channels: tuple[int, int, int, int, int] = (8, 16, 24, 32, 48)
    fusion_alpha: float = 0.05
    fusion_layers: tuple[int, int, int] = (16, 19, 22)
    num_classes: int = 80


class SharedShapePyramid(nn.Module):
    """Shared five-stage stride-2 branch producing stride 8/16/32 features."""
    def __init__(self, channels: tuple[int, int, int, int, int]) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.stages = nn.Sequential(
            ConvBNAct(1, c1, 2), ConvBNAct(c1, c2, 2), ConvBNAct(c2, c3, 2),
            ConvBNAct(c3, c4, 2), ConvBNAct(c4, c5, 2),
        )
        self.channels = channels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (2, 3, 4):
                outputs.append(x)
        return outputs[0], outputs[1], outputs[2]


class ResidualScaleFusion(nn.Module):
    def __init__(self, shape_channels: int, rgb_channels: int, alpha: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(shape_channels, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels), nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, rgb: Tensor, shape: Tensor) -> Tensor:
        if rgb.shape[-2:] != shape.shape[-2:]:
            raise ValueError(f"fusion spatial mismatch: RGB={tuple(rgb.shape)}, shape={tuple(shape.shape)}")
        return rgb + self.alpha * self.projection(shape)


class ProposedRGBShapeV2(nn.Module):
    """YOLO11n with shared Sobel shape residuals fused at P3/P4/P5."""
    def __init__(self, config: ProposedV2Config | None = None) -> None:
        super().__init__()
        self.config = config or ProposedV2Config()
        loaded = YOLO(self.config.weights, task="detect").model
        if not isinstance(loaded, DetectionModel):
            raise TypeError("weights must load an Ultralytics DetectionModel")
        if int(loaded.model[-1].nc) != self.config.num_classes:
            raise ValueError("v2 retains the pretrained 80-class detection head")
        existing = getattr(loaded, "args", {})
        loaded.args = get_cfg(overrides=existing if isinstance(existing, dict) else vars(existing))
        self.detector = loaded
        self.detect = self.detector.model[-1]
        if not isinstance(self.detect, Detect) or list(self.detect.f) != list(self.config.fusion_layers):
            raise ValueError(f"Expected YOLO11n Detect inputs {self.config.fusion_layers}")
        rgb_channels = [int(self.detect.cv2[i][0].conv.in_channels) for i in range(3)]
        self.shape_extractor = SobelMagnitude()
        self.shape_backbone = SharedShapePyramid(self.config.shape_channels)
        self.fusions = nn.ModuleList([
            ResidualScaleFusion(self.config.shape_channels[2 + i], rgb_channels[i], self.config.fusion_alpha)
            for i in range(3)
        ])
        self.last_diagnostics: dict[str, Any] = {}
        self.initialization_report = {
            "pretrained_parameter_count": sum(p.numel() for p in self.detector.parameters()),
            "new_parameter_count": sum(p.numel() for p in self.shape_backbone.parameters()) + sum(p.numel() for p in self.fusions.parameters()),
            "incompatible_keys": [], "config": asdict(self.config), "rgb_channels": rgb_channels,
        }

    @property
    def alpha_values(self) -> dict[str, float]:
        return {f"alpha{idx}": float(fusion.alpha.detach()) for idx, fusion in enumerate(self.fusions, 3)}

    def forward(self, rgb: Tensor) -> Any:
        shape_p3, shape_p4, shape_p5 = self.shape_backbone(self.shape_extractor(rgb))
        shape_features = (shape_p3, shape_p4, shape_p5)
        cache: list[Any] = []
        x: Any = rgb
        rgb_features: list[Tensor | None] = [None, None, None]
        fused_features: list[Tensor | None] = [None, None, None]
        layer_to_scale = {layer: i for i, layer in enumerate(self.config.fusion_layers)}
        for module in self.detector.model:
            if module.f != -1:
                if isinstance(module.f, int): x = cache[module.f]
                else: x = [x if index == -1 else cache[index] for index in module.f]
            x = module(x)
            if module.i in layer_to_scale:
                scale = layer_to_scale[module.i]
                rgb_features[scale] = x
                fused_features[scale] = self.fusions[scale](x, shape_features[scale])
                x = fused_features[scale]
            cache.append(x if module.i in self.detector.save else None)
        self.last_diagnostics = {
            "rgb_input": tuple(rgb.shape), "shape_input": tuple(self.shape_extractor(rgb).shape),
            "shape_p3": tuple(shape_p3.shape), "shape_p4": tuple(shape_p4.shape), "shape_p5": tuple(shape_p5.shape),
            "rgb_p3": tuple(rgb_features[0].shape), "rgb_p4": tuple(rgb_features[1].shape), "rgb_p5": tuple(rgb_features[2].shape),
            "fused_p3": tuple(fused_features[0].shape), "fused_p4": tuple(fused_features[1].shape), "fused_p5": tuple(fused_features[2].shape),
            "detection_type": type(x).__name__, "alpha": self.alpha_values,
        }
        return x

    def loss(self, batch: dict[str, Tensor], predictions: Any) -> tuple[Tensor, Any]:
        return self.detector.loss(batch, predictions)

    @torch.no_grad()
    def predict(self, images: Tensor, confidence: float = 0.001, iou: float = 0.7,
                classes: list[int] | None = None) -> list[Tensor]:
        was_training = self.training; self.eval()
        raw = self(images); prediction = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(prediction, conf_thres=confidence, iou_thres=iou,
                                         classes=classes, max_det=300, nc=self.config.num_classes)
        self.train(was_training); return detections
