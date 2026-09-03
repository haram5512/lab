from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ShapeAwareYolo


@dataclass
class RobustLossConfig:
    patch_weight: float = 1.0
    gate_weight: float = 0.25
    shape_consistency_weight: float = 0.5
    clean_detection_weight: float = 0.5
    adversarial_detection_weight: float = 1.0


class RobustTrainingLoss(nn.Module):
    """Official Ultralytics detection loss plus shape-defense objectives."""

    def __init__(
        self,
        model: ShapeAwareYolo,
        config: Optional[RobustLossConfig] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config or RobustLossConfig()

    def _detection_loss(
        self, outputs: Dict[str, object], batch: Dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        loss_vector, components = self.model.detector.loss(
            batch, outputs["predictions"]
        )
        if isinstance(components, dict):
            component_tensor = torch.stack(list(components.values()))
        else:
            component_tensor = components
        return loss_vector.sum(), component_tensor

    @staticmethod
    def _patch_and_gate_loss(
        outputs: Dict[str, object], patch_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        patch_logits = outputs["patch_logits"]
        shape_weights = outputs["shape_weights"]
        if not isinstance(patch_logits, list) or not isinstance(shape_weights, list):
            raise TypeError("Expected multi-scale patch logits and shape weights.")
        patch_losses = []
        gate_losses = []
        for logits, weights in zip(patch_logits, shape_weights):
            target = F.interpolate(
                patch_mask.float(), size=logits.shape[-2:], mode="nearest"
            )
            patch_losses.append(F.binary_cross_entropy_with_logits(logits, target))
            target_weight = 0.20 + 0.70 * target
            gate_losses.append(
                F.binary_cross_entropy(
                    weights.clamp(1e-5, 1.0 - 1e-5), target_weight
                )
            )
        return torch.stack(patch_losses).mean(), torch.stack(gate_losses).mean()

    @staticmethod
    def _shape_consistency(
        clean_outputs: Dict[str, object], adversarial_outputs: Dict[str, object]
    ) -> Tensor:
        clean_features = clean_outputs["shape_features"]
        adversarial_features = adversarial_outputs["shape_features"]
        if not isinstance(clean_features, list) or not isinstance(
            adversarial_features, list
        ):
            raise TypeError("Expected multi-scale shape features.")
        return torch.stack(
            [
                F.smooth_l1_loss(adversarial, clean.detach())
                for clean, adversarial in zip(clean_features, adversarial_features)
            ]
        ).mean()

    def forward(
        self,
        clean_outputs: Dict[str, object],
        batch: Dict[str, Tensor],
        adversarial_outputs: Optional[Dict[str, object]] = None,
        patch_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        clean_detection, clean_components = self._detection_loss(clean_outputs, batch)
        zero = clean_detection * 0.0
        adversarial_detection = zero
        adversarial_components = torch.zeros_like(clean_components)
        patch_loss = zero
        gate_loss = zero
        shape_consistency = zero

        if adversarial_outputs is not None:
            adversarial_detection, adversarial_components = self._detection_loss(
                adversarial_outputs, batch
            )
            shape_consistency = self._shape_consistency(
                clean_outputs, adversarial_outputs
            )
            if patch_mask is not None:
                patch_loss, gate_loss = self._patch_and_gate_loss(
                    adversarial_outputs, patch_mask
                )

        total = (
            self.config.clean_detection_weight * clean_detection
            + self.config.adversarial_detection_weight * adversarial_detection
            + self.config.patch_weight * patch_loss
            + self.config.gate_weight * gate_loss
            + self.config.shape_consistency_weight * shape_consistency
        )
        return {
            "total": total,
            "clean_detection": clean_detection,
            "adversarial_detection": adversarial_detection,
            "patch_localization": patch_loss,
            "gate": gate_loss,
            "shape_consistency": shape_consistency,
            "clean_yolo_components": clean_components,
            "adversarial_yolo_components": adversarial_components,
        }
