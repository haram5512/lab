"""Proposed RGB/Shape v3: YOLO11n-like full-width Shape backbone.

This variant keeps v2's P3/P4/P5 residual fusion and replaces only the
capacity-limited shape extractor.  The Shape branch mirrors YOLO11n's
Conv/C3k2/downsampling stages (with a 1-channel Sobel input) and initializes
matching layers from the pretrained RGB backbone where tensor shapes permit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
from typing import Any

import torch
from torch import Tensor, nn

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / "runs" / ".ultralytics"))
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C3k2, Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

from .proposed_rgb_shape import SobelMagnitude
from .proposed_rgb_shape_v2 import ResidualScaleFusion


@dataclass
class ProposedV3Config:
    weights: str = "yolo11n.pt"
    fusion_alpha: float = 0.05
    fusion_layers: tuple[int, int, int] = (16, 19, 22)
    num_classes: int = 80


class YOLO11LikeShapeBackbone(nn.Module):
    """YOLO11n backbone stages 0..8 adapted to one-channel Sobel input."""
    rgb_layer_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList([
            Conv(1, 16, 3, 2),
            Conv(16, 32, 3, 2),
            C3k2(32, 64, n=1, c3k=False, e=0.25),
            Conv(64, 64, 3, 2),
            C3k2(64, 128, n=1, c3k=False, e=0.25),
            Conv(128, 128, 3, 2),
            C3k2(128, 128, n=1, c3k=True, e=0.5),
            Conv(128, 256, 3, 2),
            C3k2(256, 256, n=1, c3k=True, e=0.5),
        ])

    def initialize_from_rgb(self, rgb_layers: nn.ModuleList) -> dict[str, Any]:
        copied: list[int] = []
        incompatible: list[int] = []
        with torch.no_grad():
            first_rgb = rgb_layers[0].conv.weight
            self.stages[0].conv.weight.copy_(first_rgb.mean(dim=1, keepdim=True))
            self.stages[0].bn.load_state_dict(rgb_layers[0].bn.state_dict())
            copied.append(0)
            for shape_index, rgb_index in enumerate(self.rgb_layer_indices[1:], 1):
                try:
                    result = self.stages[shape_index].load_state_dict(rgb_layers[rgb_index].state_dict(), strict=True)
                    if result.missing_keys or result.unexpected_keys:
                        incompatible.append(rgb_index)
                    else:
                        copied.append(rgb_index)
                except (RuntimeError, KeyError):
                    incompatible.append(rgb_index)
        return {"copied_rgb_layers": copied, "incompatible_rgb_layers": incompatible, "first_conv_policy": "mean over RGB input channels"}

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outputs: list[Tensor] = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (4, 6, 8):
                outputs.append(x)
        return outputs[0], outputs[1], outputs[2]


class ProposedRGBShapeV3(nn.Module):
    """YOLO11n with a full-width YOLO11-like Shape branch at P3/P4/P5."""
    def __init__(self, config: ProposedV3Config | None = None) -> None:
        super().__init__()
        self.config = config or ProposedV3Config()
        loaded = YOLO(self.config.weights, task="detect").model
        if not isinstance(loaded, DetectionModel):
            raise TypeError("weights must load an Ultralytics DetectionModel")
        if int(loaded.model[-1].nc) != self.config.num_classes:
            raise ValueError("v3 retains the pretrained 80-class detection head")
        existing = getattr(loaded, "args", {})
        loaded.args = get_cfg(overrides=existing if isinstance(existing, dict) else vars(existing))
        self.detector = loaded
        self.detect = self.detector.model[-1]
        if not isinstance(self.detect, Detect) or list(self.detect.f) != list(self.config.fusion_layers):
            raise ValueError(f"Expected YOLO11n Detect inputs {self.config.fusion_layers}")
        rgb_channels = [int(self.detect.cv2[i][0].conv.in_channels) for i in range(3)]
        self.shape_extractor = SobelMagnitude()
        self.shape_backbone = YOLO11LikeShapeBackbone()
        self.pretrained_initialization = self.shape_backbone.initialize_from_rgb(self.detector.model)
        self.fusions = nn.ModuleList([
            ResidualScaleFusion(shape_channels, rgb_channels[index], self.config.fusion_alpha)
            for index, shape_channels in enumerate((128, 128, 256))
        ])
        self.last_diagnostics: dict[str, Any] = {}
        self.initialization_report = {
            "pretrained_parameter_count": sum(parameter.numel() for parameter in self.detector.parameters()),
            "new_parameter_count": sum(parameter.numel() for parameter in self.shape_backbone.parameters()) + sum(parameter.numel() for parameter in self.fusions.parameters()),
            "incompatible_keys": self.pretrained_initialization["incompatible_rgb_layers"],
            "pretrained_shape_initialization": self.pretrained_initialization,
            "config": asdict(self.config), "rgb_channels": rgb_channels,
        }

    @property
    def alpha_values(self) -> dict[str, float]:
        return {f"alpha{index}": float(fusion.alpha.detach()) for index, fusion in enumerate(self.fusions, 3)}

    def forward(self, rgb: Tensor) -> Any:
        shape_p3, shape_p4, shape_p5 = self.shape_backbone(self.shape_extractor(rgb))
        shape_features = (shape_p3, shape_p4, shape_p5)
        cache: list[Any] = []
        x: Any = rgb
        rgb_features: list[Tensor | None] = [None, None, None]
        fused_features: list[Tensor | None] = [None, None, None]
        layer_to_scale = {layer: index for index, layer in enumerate(self.config.fusion_layers)}
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
