from __future__ import annotations

import torch
from torch import Tensor


def target_suppression_loss(raw_prediction: Tensor, target_boxes: Tensor, target_classes: Tensor, image_size: int) -> Tensor:
    """Suppress pre-NMS class confidence around selected GT objects.

    Ultralytics' decoded raw prediction is ``[B, 84, N]`` for YOLO11n:
    four box coordinates followed by 80 class scores. Anchor locations are
    selected by their decoded center, while gradients flow through scores.
    """
    if raw_prediction.ndim != 3 or raw_prediction.shape[1] < 5:
        raise ValueError(f"Unexpected raw prediction shape: {tuple(raw_prediction.shape)}")
    prediction = raw_prediction.transpose(1, 2)
    centers = prediction[..., :2]
    class_scores = prediction[..., 4:]
    losses: list[Tensor] = []
    for batch_index in range(prediction.shape[0]):
        for box, class_index in zip(target_boxes[batch_index], target_classes[batch_index]):
            cx, cy, width, height = box * image_size
            inside = (
                (centers[batch_index, :, 0] >= cx - width / 2)
                & (centers[batch_index, :, 0] <= cx + width / 2)
                & (centers[batch_index, :, 1] >= cy - height / 2)
                & (centers[batch_index, :, 1] <= cy + height / 2)
            )
            scores = class_scores[batch_index, inside, class_index.long()]
            if scores.numel() == 0:
                distance = ((centers[batch_index] - torch.stack((cx, cy))) ** 2).sum(dim=1)
                scores = class_scores[batch_index, distance.argmin(), class_index.long()].reshape(1)
            losses.append(scores.mean())
    if not losses:
        return raw_prediction.sum() * 0.0
    return torch.stack(losses).mean()

