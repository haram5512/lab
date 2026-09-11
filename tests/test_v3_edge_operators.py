from __future__ import annotations

import pytest
import torch

from adversarial_robust_detector.models.proposed_rgb_shape_v3 import (
    CannyEdge,
    LaplacianEdge,
    ProposedV3Config,
    build_edge_extractor,
)


@pytest.mark.parametrize("operator", ("sobel", "canny", "laplacian"))
def test_edge_operator_shape_range_and_finiteness(operator: str) -> None:
    extractor, metadata = build_edge_extractor(ProposedV3Config(edge_operator=operator))
    result = extractor(torch.rand(2, 3, 64, 80))
    assert result.shape == (2, 1, 64, 80)
    assert torch.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
    assert metadata["operator"] == operator


def test_canny_is_binary_and_validates_thresholds() -> None:
    result = CannyEdge()(torch.rand(1, 3, 48, 48))
    assert set(torch.unique(result).tolist()).issubset({0.0, 1.0})
    with pytest.raises(ValueError):
        CannyEdge(low_threshold=0.3, high_threshold=0.2)


def test_laplacian_constant_interior_is_zero() -> None:
    result = LaplacianEdge()(torch.ones(1, 3, 32, 32))
    assert torch.count_nonzero(result[..., 1:-1, 1:-1]) == 0
