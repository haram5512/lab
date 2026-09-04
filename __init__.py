"""Real Ultralytics YOLO with shape-aware adversarial patch defenses."""

from .losses import RobustLossConfig, RobustTrainingLoss
from .model import ModelConfig, ShapeAwareYOLO, ShapeAwareYolo
from .dataset import Coco80DetectionDataset, CocoPersonConfig, CocoPersonPatchDataset, coco_person_collate
from .baseline import BaselineDetector
from .metrics import evaluate_coco_person, inverse_letterbox_xyxy

__all__ = [
    "ModelConfig",
    "ShapeAwareYolo",
    "ShapeAwareYOLO",
    "RobustLossConfig",
    "RobustTrainingLoss",
    "CocoPersonConfig",
    "CocoPersonPatchDataset",
    "Coco80DetectionDataset",
    "coco_person_collate",
    "BaselineDetector",
    "evaluate_coco_person",
    "inverse_letterbox_xyxy",
]
