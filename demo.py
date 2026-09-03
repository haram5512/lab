"""One verified forward/backward step with the real Ultralytics YOLO head."""

import torch

from adversarial_robust_detector import ModelConfig, RobustTrainingLoss, ShapeAwareYolo


def make_batch(images: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size = images.shape[0]
    return {
        "img": images,
        "batch_idx": torch.arange(batch_size, device=images.device),
        "cls": torch.zeros(batch_size, 1, device=images.device),
        "bboxes": torch.tensor(
            [[0.50, 0.52, 0.45, 0.70]], device=images.device
        ).repeat(batch_size, 1),
    }


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeAwareYolo(ModelConfig(weights="yolo11n.pt")).to(device).train()

    clean = torch.rand(2, 3, 256, 256, device=device)
    patched = clean.clone()
    patch_mask = torch.zeros(2, 1, 256, 256, device=device)
    patch_mask[:, :, 80:144, 96:160] = 1.0
    patched[:, :, 80:144, 96:160] = torch.rand(2, 3, 64, 64, device=device)

    batch = make_batch(clean)
    criterion = RobustTrainingLoss(model)
    clean_outputs = model(clean)
    losses = criterion(clean_outputs, batch, model(patched), patch_mask)
    losses["total"].backward()

    print(f"device={device}")
    print(f"ultralytics_model={model.config.weights}, classes={model.num_classes}")
    print(
        "fused_feature_shapes=",
        [tuple(feature.shape) for feature in clean_outputs["fused_features"]],
    )
    print(
        {
            name: round(float(value.detach().mean()), 4)
            for name, value in losses.items()
        }
    )


if __name__ == "__main__":
    main()
