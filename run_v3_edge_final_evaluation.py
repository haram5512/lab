"""Full Clean/Seen/Unseen evaluation of v3 Canny and Laplacian best-seen checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from .coco_eval_core import evaluate_model
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "main_80class"
ABLATION = RUNS / "v3_edge_operator_ablation"
RUN_NAMES = {
    "sobel": "proposed_rgb_shape_v3_full",
    "canny": "proposed_v3_canny_30ep",
    "laplacian": "proposed_v3_laplacian_30ep",
}
FIELDS = (
    "person_ap", "person_ap50", "person_ap75", "person_ar100",
    "person_recall_iou50_conf025", "person_recall_iou75_conf025",
    "precision_iou50_conf025", "f1_iou50_conf025", "mean_matched_iou",
    "mean_gt_iou_with_misses", "fp_per_image_conf025", "failure_rate",
)


def paired_failure(clean_status: dict[str, list[bool]], attack_status: dict[str, list[bool]]) -> float | None:
    successes = failures = 0
    for image_id, flags in clean_status.items():
        for clean_ok, attack_ok in zip(flags, attack_status[image_id]):
            successes += int(clean_ok)
            failures += int(clean_ok and not attack_ok)
    return failures / successes if successes else None


def build_datasets(batch_size: int) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load((ROOT / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = config["validation"]["clean"]
    images = (ROOT / val["images"]).resolve()
    annotations = (ROOT / val["annotations"]).resolve()
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    categories = tuple(sorted(int(item["id"]) for item in payload["categories"]))
    common = dict(
        images_dir=images, annotations_file=annotations, image_size=640, max_images=None,
        target_policy="person", required_category_ids=(), category_ids=categories, seed=7,
        position_jitter=.015, placement_mode="torso", torso_relative_y=.38,
        patch_scale_min=.95, patch_scale_max=1.05, patch_rotation_degrees=3.,
        patch_brightness_jitter=.03, patch_perspective_jitter=.005,
    )
    # These are exactly the patch sources used by the existing Sobel v3 final evaluator.
    seen_patch = (ROOT / "runs" / "v5_diagnostic" / "patch_V5-C.png",)
    unseen_patches = (ROOT / "artifacts" / "adversarial_patches" / "unseen",)
    datasets = {
        "clean": Coco80DetectionDataset(CocoPersonConfig(
            **common, patch_mode="clean", patch_probability=0., patch_dirs=()
        )),
        "seen": Coco80DetectionDataset(CocoPersonConfig(
            **common, patch_mode="seen", patch_probability=1., patch_dirs=seen_patch
        )),
        "unseen": Coco80DetectionDataset(CocoPersonConfig(
            **common, patch_mode="unseen", patch_probability=1., patch_dirs=unseen_patches
        )),
    }
    loaders = {
        mode: DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=coco_person_collate, num_workers=0)
        for mode, dataset in datasets.items()
    }
    evaluation_config = {
        "images": str(images), "annotations": str(annotations),
        "seen_patch": str(seen_patch[0]), "unseen_patch_dir": str(unseen_patches[0]),
        "image_size": 640, "batch_size": batch_size, "seed": 7,
        "confidence": 0.001, "nms_iou": 0.7,
        "evaluator": "coco_eval_core.evaluate_model",
    }
    return annotations, datasets, loaders, evaluation_config


def evaluate_operator(operator: str, annotations: Path, datasets: dict[str, Any],
                      loaders: dict[str, Any], evaluation_config: dict[str, Any],
                      device: torch.device, output_suffix: str) -> list[dict[str, Any]]:
    run = RUNS / RUN_NAMES[operator]
    checkpoint = run / "best_seen_map.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ProposedRGBShapeV3(ProposedV3Config(
        weights=str(ROOT / "yolo11n.pt"), edge_operator=operator
    ))
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    output = run / f"final_evaluation_{output_suffix}"
    output.mkdir(parents=True, exist_ok=True)
    raw: dict[str, dict[str, Any]] = {}
    for mode in ("clean", "seen", "unseen"):
        with (output / f"{mode}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            raw[mode] = evaluate_model(
                model, loaders[mode], annotations, datasets[mode], device, 640, mode,
                confidence=.001, iou_threshold=.7,
                nms_max_time_img=float(evaluation_config["nms_max_time_img"]),
            )
    clean_status = raw["clean"]["person_match_status"]
    for mode in ("seen", "unseen"):
        raw[mode]["failure_rate"] = paired_failure(clean_status, raw[mode]["person_match_status"])
    raw["clean"]["failure_rate"] = None
    for values in raw.values():
        values.pop("person_match_status", None)
    rows = [{
        "edge_operator": operator, "checkpoint": "best_seen_map",
        "checkpoint_epoch": int(state["epoch"]), "condition": mode,
        **{field: values.get(field) for field in FIELDS},
    } for mode, values in raw.items()]
    (output / "metrics.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "edge_operator": operator, "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state["epoch"]), "selection_rule": "maximum Seen Person AP",
        "unseen_used_for_selection": False, "evaluation": evaluation_config,
    }
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("all", "sobel", "canny", "laplacian"), default="all")
    parser.add_argument("--nms-max-time-img", type=float, default=.5)
    parser.add_argument("--output-suffix", default="nms_per_image")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.nms_max_time_img <= 0 or args.batch_size < 1:
        raise ValueError("--nms-max-time-img and --batch-size must be positive")
    operators = tuple(RUN_NAMES) if args.operator == "all" else (args.operator,)
    annotations, datasets, loaders, evaluation_config = build_datasets(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_config["nms_max_time_img"] = args.nms_max_time_img
    evaluation_config["nms_time_limit_per_batch_seconds"] = 2.0 + args.nms_max_time_img * evaluation_config["batch_size"]
    rows: list[dict[str, Any]] = []
    for operator in operators:
        rows.extend(evaluate_operator(operator, annotations, datasets, loaders, evaluation_config, device, args.output_suffix))
    ABLATION.mkdir(parents=True, exist_ok=True)
    combined = ABLATION / f"final_metrics_{args.output_suffix}.csv"
    if combined.exists():
        previous = list(csv.DictReader(combined.open(encoding="utf-8")))
        rows = [row for row in previous if row["edge_operator"] not in operators] + rows
    order = {name: index for index, name in enumerate(("sobel", "canny", "laplacian"))}
    rows.sort(key=lambda row: (order[row["edge_operator"]], ("clean", "seen", "unseen").index(row["condition"])))
    with combined.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "operators": operators, "output": str(combined)}, indent=2))


if __name__ == "__main__":
    main()
