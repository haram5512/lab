from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EOTConfig:
    """Differentiable transformations sampled during patch optimization."""

    scale_min: float = 0.80
    scale_max: float = 1.20
    translate_fraction: float = 0.08
    rotation_degrees: float = 15.0
    perspective_jitter: float = 0.08
    brightness: float = 0.20
    contrast: float = 0.20


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int = 128
    patch_area_ratio: float = 0.12
    target_policy: str = "all"
    min_box_pixels: float = 16.0
    steps: int = 200
    learning_rate: float = 0.03
    seed: int = 7
    eot: EOTConfig = EOTConfig()

