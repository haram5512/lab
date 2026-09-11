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
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

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
    edge_operator: Literal["sobel", "canny", "laplacian"] = "sobel"
    canny_low_threshold: float = 0.10
    canny_high_threshold: float = 0.20
    canny_gaussian_kernel_size: int = 5
    canny_gaussian_sigma: float = 1.0
    canny_hysteresis_iterations: int = 32
    laplacian_kernel_size: int = 3


def _grayscale(rgb: Tensor) -> Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,H,W]")
    return 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


class CannyEdge(nn.Module):
    """Fixed-parameter Canny edge map implemented entirely with Torch tensors."""

    def __init__(self, low_threshold: float = 0.10, high_threshold: float = 0.20,
                 gaussian_kernel_size: int = 5, gaussian_sigma: float = 1.0,
                 hysteresis_iterations: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0.0 <= low_threshold < high_threshold <= 1.0:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 1")
        if gaussian_kernel_size < 3 or gaussian_kernel_size % 2 == 0:
            raise ValueError("Canny Gaussian kernel size must be an odd integer >= 3")
        if gaussian_sigma <= 0 or hysteresis_iterations < 1:
            raise ValueError("Canny sigma and hysteresis iterations must be positive")
        radius = gaussian_kernel_size // 2
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        gaussian_1d = torch.exp(-(axis.square()) / (2.0 * gaussian_sigma ** 2))
        gaussian_1d /= gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32
        )
        self.register_buffer("gaussian_kernel", gaussian_2d.view(1, 1, gaussian_kernel_size, gaussian_kernel_size))
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_x.t().contiguous().view(1, 1, 3, 3))
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = float(gaussian_sigma)
        self.hysteresis_iterations = hysteresis_iterations
        self.eps = eps

    def forward(self, rgb: Tensor) -> Tensor:
        gray = _grayscale(rgb)
        smoothed = F.conv2d(gray, self.gaussian_kernel, padding=self.gaussian_kernel_size // 2)
        gx = F.conv2d(smoothed, self.sobel_x, padding=1)
        gy = F.conv2d(smoothed, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + self.eps)
        magnitude = magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)

        angle = torch.remainder(torch.rad2deg(torch.atan2(gy, gx)), 180.0)
        horizontal = (angle < 22.5) | (angle >= 157.5)
        diagonal_up = (angle >= 22.5) & (angle < 67.5)
        vertical = (angle >= 67.5) & (angle < 112.5)
        diagonal_down = (angle >= 112.5) & (angle < 157.5)
        keep = (
            horizontal & (magnitude >= torch.roll(magnitude, 1, -1)) & (magnitude >= torch.roll(magnitude, -1, -1))
        ) | (
            diagonal_up & (magnitude >= torch.roll(magnitude, (1, -1), (-2, -1))) & (magnitude >= torch.roll(magnitude, (-1, 1), (-2, -1)))
        ) | (
            vertical & (magnitude >= torch.roll(magnitude, 1, -2)) & (magnitude >= torch.roll(magnitude, -1, -2))
        ) | (
            diagonal_down & (magnitude >= torch.roll(magnitude, (1, 1), (-2, -1))) & (magnitude >= torch.roll(magnitude, (-1, -1), (-2, -1)))
        )
        keep[..., 0, :] = False
        keep[..., -1, :] = False
        keep[..., :, 0] = False
        keep[..., :, -1] = False
        thinned = magnitude * keep
        strong = thinned >= self.high_threshold
        weak = thinned >= self.low_threshold
        connected = strong
        for _ in range(self.hysteresis_iterations):
            neighborhood = F.max_pool2d(connected.to(thinned.dtype), 3, stride=1, padding=1).bool()
            updated = strong | (weak & neighborhood)
            if torch.equal(updated, connected):
                break
            connected = updated
        return connected.to(rgb.dtype)


class LaplacianEdge(nn.Module):
    """Absolute 3x3 Laplacian response with per-image [0,1] normalization."""

    def __init__(self, kernel_size: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        if kernel_size != 3:
            raise ValueError("Only the fixed 3x3 Laplacian kernel is supported")
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, rgb: Tensor) -> Tensor:
        response = F.conv2d(_grayscale(rgb), self.kernel, padding=1).abs()
        maximum = response.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return response / maximum


def build_edge_extractor(config: ProposedV3Config) -> tuple[nn.Module, dict[str, Any]]:
    if config.edge_operator == "sobel":
        return SobelMagnitude(), {
            "operator": "sobel", "grayscale": "BT.601 (0.299R+0.587G+0.114B)",
            "kernel_size": 3, "magnitude": "sqrt(gx^2+gy^2+1e-6)",
            "normalization": "per-image maximum to [0,1]",
        }
    if config.edge_operator == "canny":
        return CannyEdge(
            low_threshold=config.canny_low_threshold,
            high_threshold=config.canny_high_threshold,
            gaussian_kernel_size=config.canny_gaussian_kernel_size,
            gaussian_sigma=config.canny_gaussian_sigma,
            hysteresis_iterations=config.canny_hysteresis_iterations,
        ), {
            "operator": "canny", "grayscale": "BT.601 (0.299R+0.587G+0.114B)",
            "gaussian_smoothing": True,
            "gaussian_kernel_size": config.canny_gaussian_kernel_size,
            "gaussian_sigma": config.canny_gaussian_sigma,
            "low_threshold": config.canny_low_threshold,
            "high_threshold": config.canny_high_threshold,
            "threshold_scale": "per-image normalized gradient magnitude",
            "non_maximum_suppression": "four quantized gradient directions",
            "hysteresis_iterations": config.canny_hysteresis_iterations,
            "normalization": "binary [0,1]",
        }
    if config.edge_operator == "laplacian":
        return LaplacianEdge(config.laplacian_kernel_size), {
            "operator": "laplacian", "grayscale": "BT.601 (0.299R+0.587G+0.114B)",
            "kernel": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            "kernel_size": config.laplacian_kernel_size,
            "absolute_value": True,
            "normalization": "per-image maximum to [0,1]",
        }
    raise ValueError(f"Unsupported edge operator: {config.edge_operator}")


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
        self.shape_extractor, self.edge_preprocessing = build_edge_extractor(self.config)
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
            "config": asdict(self.config), "edge_preprocessing": self.edge_preprocessing,
            "rgb_channels": rgb_channels,
        }

    @property
    def alpha_values(self) -> dict[str, float]:
        return {f"alpha{index}": float(fusion.alpha.detach()) for index, fusion in enumerate(self.fusions, 3)}

    def forward(self, rgb: Tensor) -> Any:
        shape_input = self.shape_extractor(rgb)
        shape_p3, shape_p4, shape_p5 = self.shape_backbone(shape_input)
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
            "rgb_input": tuple(rgb.shape), "shape_input": tuple(shape_input.shape),
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
                classes: list[int] | None = None, nms_max_time_img: float = 0.05) -> list[Tensor]:
        was_training = self.training; self.eval()
        raw = self(images); prediction = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(prediction, conf_thres=confidence, iou_thres=iou,
                                         classes=classes, max_det=300, nc=self.config.num_classes,
                                         max_time_img=nms_max_time_img)
        self.train(was_training); return detections
