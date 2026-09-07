"""COCO-person detection metrics for letterboxed predictions."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor


def inverse_letterbox_xyxy(
    boxes: Tensor, original_size: Tensor, scale: Tensor, pad: Tensor
) -> Tensor:
    """Map letterbox ``xyxy`` boxes back to the original image coordinates."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    width, height = original_size.tolist()
    result = boxes.clone().float()
    result[:, [0, 2]] = (result[:, [0, 2]] - pad[0]) / scale
    result[:, [1, 3]] = (result[:, [1, 3]] - pad[1]) / scale
    result[:, [0, 2]] = result[:, [0, 2]].clamp(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clamp(0, height)
    return result


def _xywh_norm_to_original(
    boxes: Tensor, original_size: Tensor, scale: Tensor, pad: Tensor, image_size: int
) -> Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, width, height = boxes.unbind(dim=1)
    letterbox = torch.stack(
        ((cx - width / 2) * image_size, (cy - height / 2) * image_size,
         (cx + width / 2) * image_size, (cy + height / 2) * image_size), dim=1
    )
    return inverse_letterbox_xyxy(letterbox, original_size, scale, pad)


def _box_iou(one: Tensor, many: Tensor) -> Tensor:
    if many.numel() == 0:
        return torch.zeros((one.shape[0], 0))
    lt = torch.maximum(one[:, None, :2], many[None, :, :2])
    rb = torch.minimum(one[:, None, 2:], many[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    area_one = ((one[:, 2] - one[:, 0]).clamp(min=0) * (one[:, 3] - one[:, 1]).clamp(min=0))[:, None]
    area_many = ((many[:, 2] - many[:, 0]).clamp(min=0) * (many[:, 3] - many[:, 1]).clamp(min=0))[None, :]
    return intersection / (area_one + area_many - intersection).clamp(min=1e-9)


def greedy_person_match_metrics(
    gt_boxes: Tensor,
    prediction_boxes: Tensor,
    prediction_scores: Tensor,
    thresholds: tuple[float, ...] = (0.50, 0.75),
) -> dict[str, Any]:
    """Score-descending, one-to-one GT matching in original-image coordinates."""
    gt_boxes = gt_boxes.reshape(-1, 4).float()
    prediction_boxes = prediction_boxes.reshape(-1, 4).float()
    prediction_scores = prediction_scores.reshape(-1).float()
    ious = _box_iou(gt_boxes, prediction_boxes)
    order = torch.argsort(prediction_scores, descending=True).tolist()
    recalls: dict[str, float] = {}
    for threshold in thresholds:
        used: set[int] = set()
        matched: list[float] = []
        for prediction_index in order:
            if not len(gt_boxes):
                break
            candidates = [(float(ious[gt_index, prediction_index]), gt_index) for gt_index in range(len(gt_boxes)) if gt_index not in used]
            if not candidates:
                continue
            overlap, gt_index = max(candidates)
            if overlap >= threshold:
                used.add(gt_index)
                matched.append(overlap)
        recalls[f"recall_iou{int(threshold * 100)}"] = len(matched) / len(gt_boxes) if len(gt_boxes) else 0.0
        if threshold == 0.50:
            matched_ious = matched
            matched_gt = used
    gt_iou_with_misses = []
    for gt_index in range(len(gt_boxes)):
        best = float(ious[gt_index].max()) if ious.shape[1] else 0.0
        gt_iou_with_misses.append(best if gt_index in matched_gt else 0.0)
    matched_tensor = torch.tensor(matched_ious, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_iou_with_misses, dtype=torch.float32)
    return {
        **recalls,
        "true_positives_iou50": int(len(matched_ious)),
        "false_positives_iou50": int(len(prediction_scores) - len(matched_ious)),
        "prediction_count": int(len(prediction_scores)),
        "matched_gt_indices_iou50": sorted(int(index) for index in matched_gt),
        "matched_ious": matched_tensor,
        "gt_iou_with_misses": gt_tensor,
        "mean_matched_iou": float(matched_tensor.mean()) if len(matched_tensor) else 0.0,
        "median_matched_iou": float(matched_tensor.median()) if len(matched_tensor) else 0.0,
        "std_matched_iou": float(matched_tensor.std(unbiased=False)) if len(matched_tensor) else 0.0,
        "matched_count": int(len(matched_tensor)),
        "mean_gt_iou_with_misses": float(gt_tensor.mean()) if len(gt_tensor) else 0.0,
    }


def _average_precision(predictions: list[tuple[int, float, Tensor]], ground_truth: dict[int, Tensor], threshold: float) -> float:
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    if total_gt == 0:
        return 0.0
    predictions = sorted(predictions, key=lambda item: item[1], reverse=True)
    matched = {image_id: torch.zeros(len(boxes), dtype=torch.bool) for image_id, boxes in ground_truth.items()}
    true_positive = []
    false_positive = []
    for image_id, score, box in predictions:
        gt = ground_truth.get(image_id, torch.empty((0, 4)))
        ious = _box_iou(box.unsqueeze(0), gt).squeeze(0)
        if ious.numel() and float(ious.max()) >= threshold:
            best = int(ious.argmax())
            if not matched[image_id][best]:
                matched[image_id][best] = True
                true_positive.append(1.0)
                false_positive.append(0.0)
                continue
        true_positive.append(0.0)
        false_positive.append(1.0)
    if not true_positive:
        return 0.0
    tp = torch.tensor(true_positive).cumsum(0)
    fp = torch.tensor(false_positive).cumsum(0)
    recall = tp / total_gt
    precision = tp / (tp + fp).clamp(min=1e-9)
    envelope = torch.flip(torch.cummax(torch.flip(precision, (0,)), dim=0).values, (0,))
    recall_points = torch.linspace(0, 1, 101)
    return float(torch.tensor([envelope[recall >= point].max() if (recall >= point).any() else 0.0 for point in recall_points]).mean())


def evaluate_coco_person(model: Any, loader: Any, device: torch.device, image_size: int) -> dict[str, float]:
    """Compute validation loss and person AP metrics for one loader pass."""
    started = time.perf_counter()
    model.eval()
    validation_losses: list[float] = []
    prediction_records: list[tuple[int, float, Tensor]] = []
    ground_truth: dict[int, Tensor] = {}
    map_started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            model_batch = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
            predictions_for_loss = model.forward_for_loss(model_batch["img"])
            loss_vector, _ = model.loss(model_batch, predictions_for_loss)
            loss = loss_vector.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")
            validation_losses.append(float(loss))
            detections = model.predict(model_batch["img"])
            for index, detection in enumerate(detections):
                image_id = int(batch["image_id"][index])
                original_size = batch["original_size"][index]
                scale = batch["letterbox_scale"][index]
                pad = batch["letterbox_pad"][index]
                ground_truth[image_id] = _xywh_norm_to_original(
                    batch["bboxes"][batch["batch_idx"] == index], original_size, scale, pad, image_size
                )
                boxes = inverse_letterbox_xyxy(detection[:, :4].cpu(), original_size, scale, pad)
                for box, confidence in zip(boxes, detection[:, 4].cpu()):
                    prediction_records.append((image_id, float(confidence), box))
    map_seconds = time.perf_counter() - map_started
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    aps = [_average_precision(prediction_records, ground_truth, threshold) for threshold in thresholds]
    return {
        "val_loss": sum(validation_losses) / len(validation_losses),
        "map": sum(aps) / len(aps),
        "ap50": aps[0],
        "ap75": aps[5],
        "val_seconds": time.perf_counter() - started,
        "map_seconds": map_seconds,
    }
