"""Run the final pretrained-vs-adversarial checkpoint evaluation matrix."""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .baseline import BaselineDetector
from .coco_eval_core import evaluate_model
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_model(weights: Path, base_weights: Path, device: torch.device) -> BaselineDetector:
    model = BaselineDetector(str(base_weights))
    if weights.resolve() != base_weights.resolve():
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError(f"Unsupported checkpoint structure: {weights}")
        model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def main() -> None:
    root = Path(__file__).resolve().parent
    config_path = root / "dataset_config.main80.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    val = config["validation"]["clean"]
    images = resolve(root, val["images"])
    annotations = resolve(root, val["annotations"])
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    all_category_ids = tuple(sorted(int(item["id"]) for item in payload["categories"]))

    output = root / "runs" / "main_80class" / "adversarial_training_final_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_weights = root / "yolo11n.pt"
    train_run = root / "runs" / "main_80class" / "v5c_adversarial_training_corrected"
    checkpoints = {
        "pretrained": base_weights,
        "best_clean": train_run / "best_clean_map.pt",
        "best_seen": train_run / "best_seen_map.pt",
    }
    for name, path in checkpoints.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing checkpoint: {name}: {path}")

    # Keep the exact v5-C training placement/transforms for Seen.  Unseen uses
    # the held-out F/G/H pool with the same placement/transformation bounds.
    patch_files = {
        "seen": (root / "runs" / "v5_diagnostic" / "patch_V5-C.png",),
        "unseen": (root / "artifacts" / "adversarial_patches" / "unseen",),
    }
    for mode, paths in patch_files.items():
        if not all(path.exists() for path in paths):
            raise FileNotFoundError(f"Missing {mode} patch source: {paths}")

    common = {
        "images_dir": images,
        "annotations_file": annotations,
        "image_size": 640,
        "max_images": None,
        "target_policy": "person",
        "required_category_ids": (),  # full COCO val image set
        "position_jitter": 0.015,
        "placement_mode": "torso",
        "torso_relative_y": 0.38,
        "patch_scale_min": 0.95,
        "patch_scale_max": 1.05,
        "patch_rotation_degrees": 3.0,
        "patch_brightness_jitter": 0.03,
        "patch_perspective_jitter": 0.005,
        "category_ids": all_category_ids,
    }
    results: dict[str, dict[str, dict]] = {name: {} for name in checkpoints}
    for model_name, checkpoint in checkpoints.items():
        model = load_model(checkpoint, base_weights, device)
        for mode in ("clean", "seen", "unseen"):
            if mode == "clean":
                dataset_cfg = CocoPersonConfig(**common, patch_mode="clean", patch_probability=0.0, patch_dirs=())
            else:
                dataset_cfg = CocoPersonConfig(
                    **common,
                    patch_mode=mode,
                    patch_probability=1.0,
                    patch_dirs=patch_files[mode],
                )
            dataset = Coco80DetectionDataset(dataset_cfg)
            loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=coco_person_collate, num_workers=0)
            log_path = logs / f"{model_name}_{mode}.log"
            with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                metrics = evaluate_model(model, loader, annotations, dataset, device, 640, mode, confidence=0.001, iou_threshold=0.7)
            metrics.update({"model": model_name, "checkpoint": str(checkpoint), "mode": mode, "images": len(dataset), "person_gt_expected": 11004})
            results[model_name][mode] = metrics
            (output / f"{model_name}_{mode}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for mode in ("clean", "seen", "unseen"):
        (output / f"{mode}_results.json").write_text(json.dumps({name: results[name][mode] for name in results}, indent=2), encoding="utf-8")

    pretrained = results["pretrained"]
    comparison: dict[str, dict] = {}
    for name in ("best_clean", "best_seen"):
        clean = results[name]["clean"]
        seen = results[name]["seen"]
        unseen = results[name]["unseen"]
        comparison[name] = {
            "clean_degradation": {
                "person_ap": pretrained["clean"]["person_ap"] - clean["person_ap"],
                "person_ap50": pretrained["clean"]["person_ap50"] - clean["person_ap50"],
                "person_ap75": pretrained["clean"]["person_ap75"] - clean["person_ap75"],
                "person_recall": pretrained["clean"]["person_recall"] - clean["person_recall"],
                "person_ar100": pretrained["clean"]["person_ar100"] - clean["person_ar100"],
            },
            "seen_absolute_improvement": {
                "person_ap": seen["person_ap"] - pretrained["seen"]["person_ap"],
                "person_ap50": seen["person_ap50"] - pretrained["seen"]["person_ap50"],
                "person_ap75": seen["person_ap75"] - pretrained["seen"]["person_ap75"],
                "person_recall": seen["person_recall"] - pretrained["seen"]["person_recall"],
                "person_ar100": seen["person_ar100"] - pretrained["seen"]["person_ar100"],
                "failure_rate_reduction": (pretrained["seen"]["seen_failure_rate"] if "seen_failure_rate" in pretrained["seen"] else 1 - pretrained["seen"]["attacked_object_detection_rate"]) - (1 - seen["attacked_object_detection_rate"] if seen["attacked_object_detection_rate"] is not None else 0),
            },
            "unseen_absolute_improvement": {
                "person_ap": unseen["person_ap"] - pretrained["unseen"]["person_ap"],
                "person_ap50": unseen["person_ap50"] - pretrained["unseen"]["person_ap50"],
                "person_ap75": unseen["person_ap75"] - pretrained["unseen"]["person_ap75"],
                "person_recall": unseen["person_recall"] - pretrained["unseen"]["person_recall"],
                "person_ar100": unseen["person_ar100"] - pretrained["unseen"]["person_ar100"],
            },
            "generalization_gap": {
                "seen_minus_unseen_ap": seen["person_ap"] - unseen["person_ap"],
                "seen_minus_unseen_recall": seen["person_recall"] - unseen["person_recall"],
            },
        }

    tradeoff = {
        "pretrained_reference": {mode: {key: pretrained[mode].get(key) for key in ("person_ap", "person_ap50", "person_ap75", "person_ar100", "person_recall", "attacked_object_detection_rate")} for mode in pretrained},
        "checkpoints": comparison,
        "recommendation": "review_clean_seen_unseen_tradeoff_before_selecting_checkpoint",
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output / "tradeoff_summary.json").write_text(json.dumps(tradeoff, indent=2), encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    metadata = {
        "git_commit": commit,
        "config": str(config_path),
        "dataset": "COCO 2017 val2017 full image set, all 80 classes with person diagnostics",
        "person_gt_expected": 11004,
        "image_size": 640,
        "confidence": 0.001,
        "nms_iou": 0.7,
        "models": {name: {"path": str(path), "sha256": sha256(path)} for name, path in checkpoints.items()},
        "patches": {mode: [str(path) for path in paths] for mode, paths in patch_files.items()},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "results": str(output),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "models": list(checkpoints), "modes": ["clean", "seen", "unseen"]}, indent=2))


if __name__ == "__main__":
    main()
