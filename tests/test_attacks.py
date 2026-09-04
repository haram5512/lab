import torch
from torch import nn

from adversarial_robust_detector.attacks import PatchConfig, PatchGenerator


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
