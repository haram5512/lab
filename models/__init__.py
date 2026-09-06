"""Research model variants."""

from .proposed_rgb_shape import ProposedRGBShapeV1, ProposedV1Config, SobelMagnitude

__all__ = ["ProposedRGBShapeV1", "ProposedV1Config", "SobelMagnitude"]
from .proposed_rgb_shape_v2 import ProposedRGBShapeV2, ProposedV2Config

__all__ = ["ProposedRGBShapeV2", "ProposedV2Config"]
