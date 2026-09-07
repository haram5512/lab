"""Unified COCO person robustness comparison for all completed model checkpoints."""
from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "runs" / ".ultralytics"))
from .baseline import BaselineDetector
from .coco_eval_core import evaluate_model
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape import ProposedRGBShapeV1, ProposedV1Config
from .models.proposed_rgb_shape_v2 import ProposedRGBShapeV2, ProposedV2Config

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "runs" / "main_80class" / "final_model_comparison"
BASE = ROOT / "yolo11n.pt"
METRIC_COLUMNS = (
    "person_ap", "person_ap50", "person_ap75", "person_ar100",
    "person_recall_iou50_conf025", "person_recall_iou75_conf025",
    "precision_iou50_conf025", "f1_iou50_conf025", "mean_matched_iou",
    "mean_gt_iou_with_misses", "fp_per_image_conf025", "failure_rate",
)


def checkpoint_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise RuntimeError(f"invalid checkpoint: {path}")
    return state


def load_model(kind: str, checkpoint: Path | None, device: torch.device) -> torch.nn.Module:
    if kind == "baseline":
        model: torch.nn.Module = BaselineDetector(str(BASE))
    elif kind == "proposed_v1":
        model = ProposedRGBShapeV1(ProposedV1Config(weights=str(BASE)))
    elif kind == "proposed_v2":
        model = ProposedRGBShapeV2(ProposedV2Config(weights=str(BASE)))
    else:
        raise ValueError(kind)
    if checkpoint is not None:
        model.load_state_dict(checkpoint_state(checkpoint)["model"], strict=True)
    return model.to(device).eval()


def model_specs() -> list[dict[str, Any]]:
    adv = ROOT / "runs" / "main_80class" / "v5c_adversarial_training_corrected"
    v1 = ROOT / "runs" / "main_80class" / "proposed_rgb_shape_v1_full"
    v2 = ROOT.parent / "runs" / "main_80class" / "proposed_rgb_shape_v2_full"
    return [
        {"model": "pretrained_yolo11n", "checkpoint": "pretrained", "kind": "baseline", "path": None},
        *[{"model": "conventional_adv", "checkpoint": name, "kind": "baseline", "path": adv / f"{name}.pt"} for name in ("best_clean_map", "best_seen_map")],
        *[{"model": "proposed_v1", "checkpoint": name, "kind": "proposed_v1", "path": v1 / f"{name}.pt"} for name in ("best_clean_map", "best_seen_map")],
        *[{"model": "proposed_v2", "checkpoint": name, "kind": "proposed_v2", "path": v2 / f"{name}.pt"} for name in ("best_clean_map", "best_seen_map")],
    ]


def build_datasets() -> tuple[dict[str, Coco80DetectionDataset], Path]:
    config = yaml.safe_load((ROOT / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = config["validation"]["clean"]
    images = (ROOT / Path(val["images"])).resolve()
    annotations = (ROOT / Path(val["annotations"])).resolve()
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    all_categories = tuple(sorted(int(item["id"]) for item in payload["categories"]))
    common: dict[str, Any] = {
        "images_dir": images, "annotations_file": annotations, "image_size": 640,
        "max_images": None, "target_policy": "person", "required_category_ids": (),
        "category_ids": all_categories, "seed": 7, "position_jitter": .015,
        "placement_mode": "torso", "torso_relative_y": .38, "patch_scale_min": .95,
        "patch_scale_max": 1.05, "patch_rotation_degrees": 3.,
        "patch_brightness_jitter": .03, "patch_perspective_jitter": .005,
    }
    patches = {
        "seen": (ROOT / "runs" / "v5_diagnostic" / "patch_V5-C.png",),
        "unseen": (ROOT / "artifacts" / "adversarial_patches" / "unseen",),
    }
    datasets: dict[str, Coco80DetectionDataset] = {
        "clean": Coco80DetectionDataset(CocoPersonConfig(**common, patch_mode="clean", patch_probability=0., patch_dirs=())),
        "seen": Coco80DetectionDataset(CocoPersonConfig(**common, patch_mode="seen", patch_probability=1., patch_dirs=patches["seen"])),
        "unseen": Coco80DetectionDataset(CocoPersonConfig(**common, patch_mode="unseen", patch_probability=1., patch_dirs=patches["unseen"])),
    }
    return datasets, annotations


def failure_rate(clean: dict[str, Any], attacked: dict[str, Any]) -> float | None:
    clean_status = clean.pop("person_match_status")
    attacked_status = attacked.pop("person_match_status")
    if clean_status.keys() != attacked_status.keys():
        return None
    clean_success = failures = 0
    for image_id, clean_matches in clean_status.items():
        attack_matches = attacked_status[image_id]
        if len(clean_matches) != len(attack_matches):
            return None
        for clean_ok, attack_ok in zip(clean_matches, attack_matches):
            clean_success += int(clean_ok)
            failures += int(clean_ok and not attack_ok)
    return failures / clean_success if clean_success else None


def latency(model: torch.nn.Module, device: torch.device) -> dict[str, float | int]:
    x = torch.rand(1, 3, 640, 640, device=device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(10): model.predict(x, confidence=.25, iou=.7, classes=None)
        if device.type == "cuda": torch.cuda.synchronize(device)
        timings = []
        for _ in range(30):
            started = time.perf_counter(); model.predict(x, confidence=.25, iou=.7, classes=None)
            if device.type == "cuda": torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
    values = torch.tensor(timings)
    median = float(values.median())
    return {"params": sum(parameter.numel() for parameter in model.parameters()), "gflops": "N/A", "forward_plus_nms_ms": median, "fps": 1000 / median, "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0}


def rows_to_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def training_summary() -> dict[str, Any]:
    run = ROOT.parent / "runs" / "main_80class" / "proposed_rgb_shape_v2_full"
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    validated = [record for record in records if record.get("seen_person_ap") is not None]
    best_clean = max(validated, key=lambda item: item["clean_person_ap"])
    best_seen = max(validated, key=lambda item: item["seen_person_ap"])
    alphas = [record["fusion_alpha"] for record in records if isinstance(record.get("fusion_alpha"), dict)]
    gradients = [float(record["shape_gradient_norm"]) for record in records if record.get("shape_gradient_norm") is not None]
    finite = all(math.isfinite(float(record["train_loss"])) for record in records)
    return {"training_completed": True, "exit_reason": "manual stop after epoch 60 validation; no meaningful Seen AP improvement", "final_epoch": records[-1]["epoch"], "early_stopping": False, "best_clean_epoch": best_clean["epoch"], "best_clean_ap": best_clean["clean_person_ap"], "best_seen_epoch": best_seen["epoch"], "best_seen_ap": best_seen["seen_person_ap"], "final_alpha": alphas[-1], "alpha_min": {key: min(item[key] for item in alphas) for key in alphas[-1]}, "alpha_max": {key: max(item[key] for item in alphas) for key in alphas[-1]}, "shape_gradient_mean": sum(gradients) / len(gradients), "shape_gradient_min": min(gradients), "shape_gradient_max": max(gradients), "nan_or_inf": not finite, "checkpoint_paths": {name: str(run / name) for name in ("best_clean_map.pt", "best_seen_map.pt", "last.pt")}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, annotations = build_datasets()
    loaders = {mode: DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=coco_person_collate, num_workers=0) for mode, dataset in datasets.items()}
    all_rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    efficiency: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for spec in model_specs():
        path = spec["path"]
        identifier = f"{spec['model']}:{spec['checkpoint']}"
        if path is not None and (not path.is_file() or path.stat().st_size == 0):
            unavailable.append({**spec, "status": "checkpoint unavailable"}); continue
        model = load_model(spec["kind"], path, device)
        if spec["checkpoint"] in ("pretrained", "best_seen_map"):
            efficiency.append({"model": spec["model"], "checkpoint": spec["checkpoint"], **latency(model, device)})
        condition_results: dict[str, dict[str, Any]] = {}
        for mode in ("clean", "seen", "unseen"):
            with (OUT / f"{spec['model']}_{spec['checkpoint']}_{mode}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                condition_results[mode] = evaluate_model(model, loaders[mode], annotations, datasets[mode], device, 640, mode, confidence=.001, iou_threshold=.7)
        for mode in ("seen", "unseen"):
            condition_results[mode]["failure_rate"] = failure_rate(deepcopy(condition_results["clean"]), condition_results[mode])
        condition_results["clean"].pop("person_match_status")
        for mode, metric in condition_results.items():
            metric.update({"model": spec["model"], "checkpoint": spec["checkpoint"], "condition": mode, "checkpoint_path": str(path) if path else str(BASE)})
            (OUT / f"{spec['model']}_{spec['checkpoint']}_{mode}.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
            all_rows.append({key: metric.get(key) for key in ("model", "checkpoint", "condition", *METRIC_COLUMNS)})
        by_id[identifier] = condition_results
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    rows_to_csv(OUT / "all_models_metrics.csv", all_rows, ["model", "checkpoint", "condition", *METRIC_COLUMNS])
    (OUT / "all_models_metrics.json").write_text(json.dumps({"results": by_id, "unavailable": unavailable}, indent=2), encoding="utf-8")
    representative = ["pretrained_yolo11n:pretrained", "conventional_adv:best_seen_map", "proposed_v1:best_seen_map", "proposed_v2:best_seen_map"]
    rep_rows = [row for row in all_rows if f"{row['model']}:{row['checkpoint']}" in representative]
    rows_to_csv(OUT / "representative_models_comparison.csv", rep_rows, ["model", "checkpoint", "condition", *METRIC_COLUMNS])
    def comparison(left: str, right: str, metrics: list[str]) -> list[dict[str, Any]]:
        rows=[]
        for mode in ("clean", "seen", "unseen"):
            for metric in metrics:
                a=by_id[left][mode].get(metric); b=by_id[right][mode].get(metric)
                rows.append({"condition":mode,"metric":metric,"left":a,"right":b,"difference_right_minus_left":(b-a) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None})
        return rows
    compared = ["person_ap", "person_recall_iou50_conf025", "person_recall_iou75_conf025", "precision_iou50_conf025", "f1_iou50_conf025", "mean_gt_iou_with_misses", "failure_rate"]
    rows_to_csv(OUT / "proposed_v2_vs_adv.csv", comparison("conventional_adv:best_seen_map", "proposed_v2:best_seen_map", compared), ["condition", "metric", "left", "right", "difference_right_minus_left"])
    rows_to_csv(OUT / "v1_vs_v2.csv", comparison("proposed_v1:best_seen_map", "proposed_v2:best_seen_map", compared), ["condition", "metric", "left", "right", "difference_right_minus_left"])
    degradation=[]
    for identifier, values in by_id.items():
        clean=values["clean"]
        for mode in ("seen", "unseen"):
            for metric in ("person_ap","person_ap50","person_ap75","person_recall_iou50_conf025","person_recall_iou75_conf025","precision_iou50_conf025","f1_iou50_conf025","mean_gt_iou_with_misses"):
                degradation.append({"model_checkpoint":identifier,"attack_condition":mode,"metric":metric,"clean_minus_attack":clean[metric]-values[mode][metric]})
    rows_to_csv(OUT / "degradation_analysis.csv", degradation, ["model_checkpoint","attack_condition","metric","clean_minus_attack"])
    summary=training_summary(); (OUT / "alpha_gradient_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows_to_csv(OUT / "efficiency_comparison.csv", efficiency, ["model","checkpoint","params","gflops","forward_plus_nms_ms","fps","peak_vram_gib"])
    adv=by_id["conventional_adv:best_seen_map"]; v2=by_id["proposed_v2:best_seen_map"]
    unseen_improvement = all(v2["unseen"][name] > adv["unseen"][name] for name in ("person_ap","person_recall_iou50_conf025","f1_iou50_conf025","mean_gt_iou_with_misses"))
    verdict = "GO" if unseen_improvement else "NO-GO"
    report = f"# Final Model Comparison\n\nFinal verdict: **{verdict}**.\n\nThe machine-readable tables in this directory are the authoritative record. Evaluation uses COCO val2017 person-containing images (11,004 person GT), 640 letterbox/inverse-letterbox, COCOeval, confidence 0.25 for operating-point Recall/Precision/F1, and one-to-one greedy person matching.\n\nProposed v2 training stopped manually after epoch {summary['final_epoch']} because Seen AP did not meaningfully exceed its previous best.\n"
    (OUT / "FINAL_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    metadata={"git_commit":subprocess.check_output(["git","-c",f"safe.directory={ROOT}","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU","image_size":640,"evaluation_confidence":.001,"nms_iou":.7,"operating_point_confidence":.25,"person_gt_expected":11004,"unavailable":unavailable}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output":str(OUT),"models_evaluated":len(by_id),"verdict":verdict}, indent=2))

if __name__ == "__main__":
    main()
