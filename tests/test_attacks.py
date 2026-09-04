import torch
from torch import nn

from adversarial_robust_detector.attacks import PatchConfig, PatchGenerator
from adversarial_robust_detector.attacks.patch_losses import gt_matched_topk_suppression_loss


class TinyRawDetector(nn.Module):
    """Differentiable pre-NMS-shaped detector fixture for attack tests."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor_bias = nn.Parameter(torch.zeros(()))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch = image.shape[0]
        anchors = 4
        boxes = torch.tensor(
            [[[320.0, 320.0, 160.0, 160.0]] * anchors],
            device=image.device,
            dtype=image.dtype,
        ).expand(batch, anchors, 4)
        scores = image.mean(dim=(1, 2, 3)).reshape(batch, 1, 1).expand(batch, 80, anchors)
        return torch.cat((boxes.transpose(1, 2), scores), dim=1), {}


def test_patch_updates_while_detector_stays_frozen() -> None:
    detector = TinyRawDetector()
    generator = PatchGenerator(detector, PatchConfig(patch_size=32, steps=2, learning_rate=0.1), device=torch.device("cpu"))
    image = torch.full((1, 3, 640, 640), 0.5)
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])
    classes = torch.zeros((1, 1))
    result = generator.generate(image, boxes, classes)
    assert len(result.losses) == 2
    assert torch.all((result.patch >= 0) & (result.patch <= 1))
    assert all(parameter.grad is None for parameter in detector.parameters())
    assert result.patch.std() > 0


def test_v3_gt_matched_loss_is_scalar_finite_and_differentiable() -> None:
    raw = torch.zeros((1, 84, 4), requires_grad=True)
    raw.data[:, :4, :] = torch.tensor([[[320.0], [320.0], [160.0], [160.0]]])
    raw.data[:, 4, :] = torch.tensor([0.9, 0.7, 0.1, 0.01])
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])
    classes = torch.zeros((1, 1))
    loss = gt_matched_topk_suppression_loss(raw, boxes, classes, 640, top_k=2)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None and raw.grad[:, 4].norm() > 0


def test_v3_handles_multiple_gt_and_no_overlap() -> None:
    raw = torch.zeros((1, 84, 3), requires_grad=True)
    raw.data[:, :4, :] = torch.tensor([[[10.0], [10.0], [4.0], [4.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]])
    classes = torch.zeros((1, 2))
    loss = gt_matched_topk_suppression_loss(raw, boxes, classes, 640)
    loss.backward()
    assert torch.isfinite(loss) and raw.grad is not None


def test_v4_shared_patch_receives_multi_image_gradient() -> None:
    detector = TinyRawDetector()
    config = PatchConfig(patch_size=16, steps=2, learning_rate=0.05, objective="v3")
    generator = PatchGenerator(detector, config, device=torch.device("cpu"))
    images = [torch.full((1, 3, 640, 640), value) for value in (0.3, 0.7)]
    boxes = [torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]), torch.tensor([[[0.5, 0.5, 0.25, 0.25], [0.6, 0.6, 0.1, 0.1]]])]
    classes = [torch.zeros((1, 1)), torch.zeros((1, 2))]
    result = generator.generate_many(images, boxes, classes, batch_size=2)
    assert len(result.gradient_norms) == 2
    assert all(torch.isfinite(torch.tensor(result.gradient_norms)))
    assert all(value > 0 for value in result.gradient_norms)


def test_torso_placement_is_inside_bbox_and_above_center() -> None:
    image = torch.zeros((1, 3, 100, 100))
    patch = torch.ones((1, 3, 16, 16))
    box = torch.tensor([0.5, 0.5, 0.4, 0.6])
    center = PatchGenerator(TinyRawDetector(), PatchConfig(placement_mode="center"), torch.device("cpu"))._place(patch, image, box)
    torso = PatchGenerator(TinyRawDetector(), PatchConfig(placement_mode="torso"), torch.device("cpu"))._place(patch, image, box)
    center_rows = torch.nonzero(center[0, 0].sum(1), as_tuple=False).flatten()
    torso_rows = torch.nonzero(torso[0, 0].sum(1), as_tuple=False).flatten()
    assert torso_rows.float().mean() < center_rows.float().mean()
    assert torso_rows.min() >= 20 and torso_rows.max() <= 80
