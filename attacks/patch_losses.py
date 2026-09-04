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


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:4]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def gt_matched_topk_suppression_loss(
    raw_prediction: Tensor,
    target_boxes: Tensor,
    target_classes: Tensor,
    image_size: int,
    iou_threshold: float = 0.10,
    top_k: int = 10,
) -> Tensor:
    """Minimize high-score raw candidates that spatially match each GT.

    YOLO11 inference output is decoded ``[B, 4 + nc, N]``; there is no
    separate objectness channel. Candidate selection uses detached decoded
    boxes and scores, while the selected person scores retain gradients.
    """
    if raw_prediction.ndim != 3 or raw_prediction.shape[1] < 5:
        raise ValueError(f"Unexpected raw prediction shape: {tuple(raw_prediction.shape)}")
    prediction = raw_prediction.transpose(1, 2)
    candidate_boxes = _xywh_to_xyxy(prediction[..., :4])
    class_scores = prediction[..., 4:]
    losses: list[Tensor] = []
    for batch_index in range(prediction.shape[0]):
        for gt, class_index in zip(target_boxes[batch_index], target_classes[batch_index]):
            gt_xywh = gt * image_size
            gt_xyxy = _xywh_to_xyxy(gt_xywh.reshape(1, 4))[0]
            boxes = candidate_boxes[batch_index]
            lt = torch.maximum(boxes[:, :2].detach(), gt_xyxy[:2])
            rb = torch.minimum(boxes[:, 2:].detach(), gt_xyxy[2:])
            wh = (rb - lt).clamp(min=0)
            intersection = wh[:, 0] * wh[:, 1]
            gt_area = (gt_xyxy[2] - gt_xyxy[0]).clamp(min=0) * (gt_xyxy[3] - gt_xyxy[1]).clamp(min=0)
            box_area = (boxes[:, 2].detach() - boxes[:, 0].detach()).clamp(min=0) * (boxes[:, 3].detach() - boxes[:, 1].detach()).clamp(min=0)
            iou = intersection / (gt_area + box_area - intersection).clamp(min=1e-9)
            scores = class_scores[batch_index, :, int(class_index.item())]
            eligible = torch.nonzero(iou >= iou_threshold, as_tuple=False).flatten()
            if eligible.numel() == 0:
                eligible = torch.topk(iou, k=min(top_k, iou.numel())).indices
            selected_scores = scores[eligible]
            selected_scores = torch.topk(selected_scores, k=min(top_k, selected_scores.numel())).values
            losses.append(selected_scores.mean())
    return torch.stack(losses).mean() if losses else raw_prediction.sum() * 0.0
