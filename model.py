from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Keep Ultralytics settings inside the project instead of writing to AppData.
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent)
)

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression


@dataclass
class ModelConfig:
    """Configuration for a real Ultralytics YOLO with shape-aware fusion."""

    weights: str = "yolo11n.pt"
    num_classes: Optional[int] = None
    shape_base_channels: int = 32
    disagreement_gain: float = 2.0
    clean_shape_prior: float = 0.20
    freeze_yolo: bool = False


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 2)
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, channels, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ShapePreprocessor(nn.Module):
    """Create grayscale, Sobel-magnitude and Laplacian shape channels."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    @staticmethod
    def _normalize_per_image(x: Tensor) -> Tensor:
        return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W].")
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        laplacian = F.conv2d(gray, self.laplacian, padding=1).abs()
        return torch.cat(
            (
                gray,
                self._normalize_per_image(magnitude),
                self._normalize_per_image(laplacian),
            ),
            dim=1,
        )


class ShapePyramid(nn.Module):
    """Independent shape encoder that returns stride 8, 16 and 32 features."""

    def __init__(self, output_channels: Sequence[int], base_channels: int) -> None:
        super().__init__()
        if len(output_channels) != 3:
            raise ValueError("This implementation expects three YOLO detection scales.")
        b = max(8, base_channels)
        self.stem = ConvBNAct(3, b, 3, 2)
        self.stage2 = nn.Sequential(ConvBNAct(b, b * 2, 3, 2), ResidualBlock(b * 2))
        self.stage3 = nn.Sequential(ConvBNAct(b * 2, b * 4, 3, 2), ResidualBlock(b * 4))
        self.stage4 = nn.Sequential(ConvBNAct(b * 4, b * 8, 3, 2), ResidualBlock(b * 8))
        self.stage5 = nn.Sequential(ConvBNAct(b * 8, b * 8, 3, 2), ResidualBlock(b * 8))
        self.projections = nn.ModuleList(
            (
                ConvBNAct(b * 4, output_channels[0], 1),
                ConvBNAct(b * 8, output_channels[1], 1),
                ConvBNAct(b * 8, output_channels[2], 1),
            )
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return [
            projection(feature)
            for projection, feature in zip(self.projections, (p3, p4, p5))
        ]


class DisagreementAdaptiveFusion(nn.Module):
    """Increase local shape reliance where RGB and shape features disagree."""

    def __init__(
        self,
        channels: int,
        disagreement_gain: float,
        clean_shape_prior: float,
    ) -> None:
        super().__init__()
        prior = min(max(clean_shape_prior, 1e-4), 1.0 - 1e-4)
        self.register_buffer("clean_shape_prior_logit", torch.logit(torch.tensor(prior)))
        self.disagreement_gain = disagreement_gain
        comparison_channels = channels * 3 + 1
        self.patch_head = nn.Sequential(
            ConvBNAct(comparison_channels, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )
        self.gate_head = nn.Sequential(
            ConvBNAct(comparison_channels, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )
        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, 3), ResidualBlock(channels)
        )

    def forward(self, rgb: Tensor, shape: Tensor) -> Dict[str, Tensor]:
        if shape.shape[-2:] != rgb.shape[-2:]:
            shape = F.interpolate(
                shape, size=rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        rgb_norm = F.normalize(rgb, p=2, dim=1, eps=1e-6)
        shape_norm = F.normalize(shape, p=2, dim=1, eps=1e-6)
        difference = (rgb_norm - shape_norm).abs()
        disagreement = difference.mean(dim=1, keepdim=True)
        comparison = torch.cat((rgb, shape, difference, disagreement), dim=1)
        patch_logits = self.patch_head(comparison)
        learned_gate = self.gate_head(comparison)
        shape_weight = torch.sigmoid(
            self.clean_shape_prior_logit
            + learned_gate
            + self.disagreement_gain * patch_logits
        )
        fused = (1.0 - shape_weight) * rgb + shape_weight * shape
        return {
            "fused": self.refine(fused),
            "disagreement": disagreement,
            "patch_logits": patch_logits,
            "shape_weight": shape_weight,
        }


class ShapeAwareYolo(nn.Module):
    """Ultralytics YOLO with shape-aware P3/P4/P5 inputs to Detect.

    The official YOLO graph, Detect head, Task-Aligned Assigner and detection
    losses are retained. Only the three tensors immediately before Detect are
    replaced by disagreement-guided RGB/shape mixtures.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.detector = self._load_detection_model(self.config)
        if self.config.freeze_yolo:
            for parameter in self.detector.parameters():
                parameter.requires_grad_(False)

        self.detect = self.detector.model[-1]
        if not isinstance(self.detect, Detect):
            raise TypeError("The selected Ultralytics model does not end with Detect.")
        detection_channels = self._detect_input_channels(self.detect)
        self.shape_preprocessor = ShapePreprocessor()
        self.shape_pyramid = ShapePyramid(
            detection_channels, self.config.shape_base_channels
        )
        self.fusion = nn.ModuleList(
            DisagreementAdaptiveFusion(
                channels,
                self.config.disagreement_gain,
                self.config.clean_shape_prior,
            )
            for channels in detection_channels
        )
        self.num_classes = int(self.detect.nc)

    @staticmethod
    def _load_detection_model(config: ModelConfig) -> DetectionModel:
        source = config.weights
        if config.num_classes is None:
            model = YOLO(source, task="detect").model
            if not isinstance(model, DetectionModel):
                raise TypeError(f"{source!r} is not an Ultralytics detection model.")
            return ShapeAwareYolo._attach_training_args(model)

        if str(source).lower().endswith(".pt"):
            pretrained = YOLO(source, task="detect").model
            if not isinstance(pretrained, DetectionModel):
                raise TypeError(f"{source!r} is not an Ultralytics detection model.")
            model = DetectionModel(
                deepcopy(pretrained.yaml), ch=3, nc=config.num_classes, verbose=False
            )
            model.load(pretrained)
            return ShapeAwareYolo._attach_training_args(model)
        model = DetectionModel(source, ch=3, nc=config.num_classes, verbose=False)
        return ShapeAwareYolo._attach_training_args(model)

    @staticmethod
    def _attach_training_args(model: DetectionModel) -> DetectionModel:
        """Attach attribute-style Ultralytics hyperparameters required by loss."""

        existing = getattr(model, "args", {})
        if not isinstance(existing, dict):
            existing = vars(existing)
        model.args = get_cfg(overrides=existing)
        return model

    @staticmethod
    def _detect_input_channels(detect: Detect) -> Tuple[int, int, int]:
        channels = tuple(int(branch[0].conv.in_channels) for branch in detect.cv2)
        if len(channels) != 3:
            raise ValueError(
                f"Expected a three-scale Detect head, found {len(channels)} scales."
            )
        return channels  # type: ignore[return-value]

    def _run_yolo_graph(
        self, image: Tensor, shape_features: Sequence[Tensor]
    ) -> Tuple[object, List[Dict[str, Tensor]]]:
        cache: List[Optional[object]] = []
        x: object = image
        fusion_outputs: List[Dict[str, Tensor]] = []

        for module in self.detector.model:
            if module.f != -1:
                if isinstance(module.f, int):
                    x = cache[module.f]
                else:
                    x = [x if index == -1 else cache[index] for index in module.f]

            if module is self.detect:
                if not isinstance(x, (list, tuple)) or len(x) != len(self.fusion):
                    raise RuntimeError("Unexpected feature inputs at YOLO Detect.")
                fusion_outputs = [
                    fusion(rgb_feature, shape_feature)
                    for fusion, rgb_feature, shape_feature in zip(
                        self.fusion, x, shape_features
                    )
                ]
                x = module([item["fused"] for item in fusion_outputs])
            else:
                x = module(x)
            cache.append(x if module.i in self.detector.save else None)
        return x, fusion_outputs

    def forward(
        self, rgb: Tensor, shape_input: Optional[Tensor] = None
    ) -> Dict[str, object]:
        if rgb.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ):
            raise TypeError("rgb must be a floating-point tensor normalized to [0, 1].")
        if shape_input is None:
            shape_input = self.shape_preprocessor(rgb)
        shape_features = self.shape_pyramid(shape_input)
        predictions, fusion_outputs = self._run_yolo_graph(rgb, shape_features)
        return {
            "predictions": predictions,
            "shape_input": shape_input,
            "shape_features": shape_features,
            "fused_features": [item["fused"] for item in fusion_outputs],
            "disagreement_maps": [item["disagreement"] for item in fusion_outputs],
            "patch_logits": [item["patch_logits"] for item in fusion_outputs],
            "shape_weights": [item["shape_weight"] for item in fusion_outputs],
        }

    @torch.no_grad()
    def predict_tensor(
        self,
        images: Tensor,
        confidence: float = 0.25,
        iou: float = 0.7,
        max_detections: int = 300,
    ) -> List[Tensor]:
        """Run NMS on a normalized BCHW tensor; return xyxy/conf/class rows."""

        was_training = self.training
        self.eval()
        raw = self(images)["predictions"]
        if isinstance(raw, tuple):
            raw = raw[0]
        if isinstance(raw, dict):
            raw = raw.get("one2many", raw)
        detections = non_max_suppression(
            raw,
            conf_thres=confidence,
            iou_thres=iou,
            max_det=max_detections,
            nc=self.num_classes,
        )
        self.train(was_training)
        return detections


ShapeAwareYOLO = ShapeAwareYolo
