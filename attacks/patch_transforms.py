from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .patch_config import EOTConfig


def transform_patch(patch: Tensor, config: EOTConfig, generator: torch.Generator) -> Tensor:
    """Apply differentiable EOT scale/translation/rotation/appearance jitter.

    The perspective term is implemented as a bounded shear/scale jitter in the
    affine grid. This keeps the first attack implementation differentiable and
    stable; a full homography can be added behind the same interface later.
    """
    batch, _, height, width = patch.shape
    device = patch.device
    scale = torch.empty((), device=device).uniform_(config.scale_min, config.scale_max, generator=generator)
    angle = torch.empty((), device=device).uniform_(-config.rotation_degrees, config.rotation_degrees, generator=generator) * math.pi / 180.0
    tx = torch.empty((), device=device).uniform_(-config.translate_fraction, config.translate_fraction, generator=generator)
    ty = torch.empty((), device=device).uniform_(-config.translate_fraction, config.translate_fraction, generator=generator)
    shear = torch.empty((), device=device).uniform_(-config.perspective_jitter, config.perspective_jitter, generator=generator)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    matrix = torch.zeros((batch, 2, 3), device=device, dtype=patch.dtype)
    matrix[:, 0, 0] = cosine / scale
    matrix[:, 0, 1] = -sine + shear
    matrix[:, 1, 0] = sine
    matrix[:, 1, 1] = cosine / scale
    matrix[:, :, 2] = torch.stack((tx, ty)).expand(batch, 2)
    grid = F.affine_grid(matrix, patch.shape, align_corners=False)
    transformed = F.grid_sample(patch, grid, mode="bilinear", padding_mode="border", align_corners=False)
    brightness = torch.empty((), device=device).uniform_(1.0 - config.brightness, 1.0 + config.brightness, generator=generator)
    contrast = torch.empty((), device=device).uniform_(1.0 - config.contrast, 1.0 + config.contrast, generator=generator)
    return ((transformed - 0.5) * contrast + 0.5) * brightness

