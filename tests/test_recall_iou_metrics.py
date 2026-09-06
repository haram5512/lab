import torch

from adversarial_robust_detector.metrics import greedy_person_match_metrics


def test_perfect_and_empty_matching() -> None:
    gt = torch.tensor([[0., 0., 10., 10.]])
    pred = torch.tensor([[0., 0., 10., 10.]])
    result = greedy_person_match_metrics(gt, pred, torch.tensor([.9]))
    assert result["recall_iou50"] == 1.0
    assert result["recall_iou75"] == 1.0
    assert result["matched_ious"].tolist() == [1.0]
    empty = greedy_person_match_metrics(gt, torch.empty((0, 4)), torch.empty(0))
    assert empty["recall_iou50"] == 0.0
    assert empty["mean_gt_iou_with_misses"] == 0.0


def test_partial_overlap_thresholds_and_gt_misses() -> None:
    gt = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])
    pred = torch.tensor([[0., 0., 8., 10.], [20., 20., 25., 30.]])
    result = greedy_person_match_metrics(gt, pred, torch.tensor([.9, .8]))
    assert result["recall_iou50"] == 1.0
    assert result["recall_iou75"] == 0.5
    assert result["matched_count"] == 2
    assert 0.0 < result["mean_matched_iou"] < 1.0
    assert result["mean_gt_iou_with_misses"] == result["mean_matched_iou"]


def test_score_descending_one_to_one_matching() -> None:
    gt = torch.tensor([[0., 0., 10., 10.], [0., 0., 8., 8.]])
    pred = torch.tensor([[0., 0., 8., 8.], [0., 0., 10., 10.]])
    result = greedy_person_match_metrics(gt, pred, torch.tensor([.95, .5]))
    assert result["recall_iou75"] == 1.0
    assert result["matched_count"] == 2
