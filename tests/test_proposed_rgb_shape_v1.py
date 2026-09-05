from pathlib import Path

import torch

from adversarial_robust_detector.models.proposed_rgb_shape import (
    ProposedRGBShapeV1,
    ProposedV1Config,
    SobelMagnitude,
)


def make_model() -> ProposedRGBShapeV1:
    return ProposedRGBShapeV1(ProposedV1Config(weights="yolo11n.pt", fusion_alpha=0.05))


def test_sobel_shape_and_normalization_are_finite() -> None:
    shape = SobelMagnitude()(torch.rand(2, 3, 128, 128))
    assert shape.shape == (2, 1, 128, 128)
    assert torch.isfinite(shape).all()
    assert float(shape.min()) >= 0.0 and float(shape.max()) <= 1.0


def test_p4_shapes_and_detection_contract() -> None:
    model = make_model().eval()
    output = model(torch.rand(1, 3, 128, 128))
    diagnostics = model.last_diagnostics
    assert diagnostics["rgb_p4"] == (1, 128, 8, 8)
    assert diagnostics["shape_p4"] == (1, 32, 8, 8)
    assert diagnostics["fused_p4"] == (1, 128, 8, 8)
    prediction = output[0] if isinstance(output, tuple) else output
    assert prediction.ndim == 3 and prediction.shape[1] == 84
    detections = model.predict(torch.rand(1, 3, 128, 128), confidence=0.99)
    assert detections[0].shape[1] == 6


def test_multi_person_clean_perturbed_backward_has_shape_and_fusion_gradients() -> None:
    model = make_model().train()
    clean = torch.rand(2, 3, 128, 128)
    perturbed = clean.clone()
    perturbed[:, :, 40:72, 48:80] = torch.rand(2, 3, 32, 32)
    batch = {
        "img": clean,
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([[0.0], [0.0], [0.0]]),
        "bboxes": torch.tensor([[.4, .5, .2, .5], [.7, .5, .15, .4], [.5, .5, .3, .6]]),
    }
    clean_loss, _ = model.loss(batch, model(clean))
    batch["img"] = perturbed
    perturbed_loss, _ = model.loss(batch, model(perturbed))
    loss = clean_loss.sum() + perturbed_loss.sum()
    loss.backward()
    shape_norm = sum(float(p.grad.norm()) for p in model.shape_backbone.parameters() if p.grad is not None)
    fusion_norm = sum(float(p.grad.norm()) for p in model.fusion.parameters() if p.grad is not None)
    rgb_norm = sum(float(p.grad.norm()) for p in model.detector.model[0].parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert shape_norm > 0 and fusion_norm > 0 and rgb_norm > 0


def test_pretrained_initialization_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = make_model()
    report = model.initialization_report
    assert report["pretrained_parameter_count"] > 0
    assert report["new_parameter_count"] > 0
    assert report["incompatible_keys"] == []
    checkpoint = tmp_path / "proposed_v1.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    restored = make_model()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
