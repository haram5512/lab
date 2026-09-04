"""Gradient-based adversarial patch generation utilities."""

from .patch_config import EOTConfig, PatchConfig
from .patch_generator import PatchGenerator, PatchResult

__all__ = ["EOTConfig", "PatchConfig", "PatchGenerator", "PatchResult"]
