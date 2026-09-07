"""Research model variants."""

from .proposed_rgb_shape_v2 import ProposedRGBShapeV2, ProposedV2Config
from .proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config
from .proposed_rgb_shape import ProposedRGBShapeV1, ProposedV1Config, SobelMagnitude

__all__ = [
    "ProposedRGBShapeV1", "ProposedV1Config", "SobelMagnitude",
    "ProposedRGBShapeV2", "ProposedV2Config",
    "ProposedRGBShapeV3", "ProposedV3Config",
]
