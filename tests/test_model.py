import torch

from adversarial_robust_detector import ModelConfig, RobustTrainingLoss, ShapeAwareYolo


def make_batch(images: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "img": images,
        "batch_idx": torch.tensor([0], device=images.device),
        "cls": torch.tensor([[0.0]], device=images.device),
        "bboxes": torch.tensor(
            [[0.50, 0.52, 0.45, 0.70]], device=images.device
        ),
    }


def make_model() -> ShapeAwareYolo:
    return ShapeAwareYolo(
        ModelConfig(
            weights="yolo11n.yaml",
            num_classes=2,
            shape_base_channels=8,
        )
    )


def test_uses_real_ultralytics_detect_and_fuses_three_scales() -> None:
    model = make_model().train()
    outputs = model(torch.rand(1, 3, 128, 128))
    fused = outputs["fused_features"]
    assert model.detect.__class__.__module__.startswith("ultralytics.")
    assert [item.shape[-2:] for item in fused] == [(16, 16), (8, 8), (4, 4)]
    assert model.num_classes == 2
    assert all(weight.shape[1] == 1 for weight in outputs["shape_weights"])


def test_official_yolo_loss_and_custom_losses_backward() -> None:
    model = make_model().train()
    clean = torch.rand(1, 3, 128, 128)
    patched = clean.clone()
    patched[:, :, 40:72, 48:80] = torch.rand(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 128, 128)
    mask[:, :, 40:72, 48:80] = 1

    criterion = RobustTrainingLoss(model)
    losses = criterion(model(clean), make_batch(clean), model(patched), mask)
    losses["total"].backward()

    assert torch.isfinite(losses["total"])
    assert losses["clean_yolo_components"].numel() >= 3
    assert any(parameter.grad is not None for parameter in model.fusion.parameters())


def test_eval_nms_returns_standard_yolo_rows() -> None:
    model = make_model().eval()
    detections = model.predict_tensor(torch.rand(1, 3, 128, 128), confidence=0.99)
    assert len(detections) == 1
    assert detections[0].ndim == 2
    assert detections[0].shape[1] == 6
