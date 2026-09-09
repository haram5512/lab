"""Full fixed-alpha ablation for Proposed RGB/Shape v3.

The same checkpoint is evaluated with its stored fusion alphas and with only
the inference-time alpha contribution forced to zero.  No checkpoint, model
weights, evaluator definitions, or attack settings are modified on disk.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "runs" / ".ultralytics"))

import torch
import yaml
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from metrics import inverse_letterbox_xyxy
from models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "runs" / "main_80class" / "v3_rgb_only_vs_shape_ablation"
CHECKPOINT = ROOT.parent / "runs" / "main_80class" / "proposed_rgb_shape_v3_full" / "best_seen_map.pt"
IMAGE_SIZE = 640
CONFIDENCE = 0.001
PERSON_CONFIDENCE = 0.25
NMS_IOU = 0.7


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def box_iou(one: torch.Tensor, many: torch.Tensor) -> torch.Tensor:
    if one.numel() == 0 or many.numel() == 0:
        return torch.zeros((one.reshape(-1, 4).shape[0], many.reshape(-1, 4).shape[0]))
    lt = torch.maximum(one[:, None, :2], many[None, :, :2])
    rb = torch.minimum(one[:, None, 2:], many[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((one[:, 2] - one[:, 0]).clamp(min=0) * (one[:, 3] - one[:, 1]).clamp(min=0))[:, None]
    area2 = ((many[:, 2] - many[:, 0]).clamp(min=0) * (many[:, 3] - many[:, 1]).clamp(min=0))[None]
    return inter / (area1 + area2 - inter).clamp_min(1e-9)


def greedy_mapping(gt: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> dict[str, Any]:
    ious = box_iou(gt, boxes)
    used: set[int] = set()
    per_gt = [{"success": False, "iou": 0.0, "confidence": 0.0, "prediction_index": None} for _ in range(len(gt))]
    for prediction_index in torch.argsort(scores, descending=True).tolist():
        candidates = [(float(ious[g, prediction_index]), g) for g in range(len(gt)) if g not in used]
        if not candidates:
            continue
        overlap, gt_index = max(candidates)
        if overlap >= threshold:
            used.add(gt_index)
            per_gt[gt_index] = {"success": True, "iou": overlap,
                                "confidence": float(scores[prediction_index]),
                                "prediction_index": int(prediction_index)}
    return {"per_gt": per_gt, "matched": len(used), "predictions": len(boxes)}


def original_gt(batch: dict[str, Any], index: int) -> torch.Tensor:
    mask = batch["batch_idx"] == index
    boxes = batch["bboxes"][mask]
    classes = batch["cls"][mask].reshape(-1)
    boxes = boxes[classes == 0]
    if not len(boxes):
        return torch.empty((0, 4))
    x, y, w, h = boxes.unbind(1)
    letterbox = torch.stack(((x-w/2)*IMAGE_SIZE, (y-h/2)*IMAGE_SIZE,
                             (x+w/2)*IMAGE_SIZE, (y+h/2)*IMAGE_SIZE), 1)
    return inverse_letterbox_xyxy(letterbox, batch["original_size"][index],
                                  batch["letterbox_scale"][index], batch["letterbox_pad"][index])


def empty_accumulator() -> dict[str, Any]:
    return {"coco_predictions": [], "person_gt": 0, "person_predictions": 0,
            "matched50": 0, "matched75": 0, "matched_ious": [],
            "gt_ious": [], "statuses": {}}


def add_prediction(acc: dict[str, Any], detection: torch.Tensor, restored: torch.Tensor,
                   gt: torch.Tensor, image_id: int, class_to_category: dict[int, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    det_cpu = detection.detach().cpu()
    for box, score, class_index in zip(restored, det_cpu[:, 4], det_cpu[:, 5].long()):
        x1, y1, x2, y2 = map(float, box)
        acc["coco_predictions"].append({"image_id": image_id,
            "category_id": int(class_to_category[int(class_index)]),
            "bbox": [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)], "score": float(score)})
    person_mask = (det_cpu[:, 5].long() == 0) & (det_cpu[:, 4] >= PERSON_CONFIDENCE)
    boxes = restored[person_mask]
    scores = det_cpu[:, 4][person_mask]
    m50 = greedy_mapping(gt, boxes, scores, 0.50)
    m75 = greedy_mapping(gt, boxes, scores, 0.75)
    acc["person_gt"] += len(gt); acc["person_predictions"] += len(boxes)
    acc["matched50"] += m50["matched"]; acc["matched75"] += m75["matched"]
    acc["matched_ious"].extend(x["iou"] for x in m50["per_gt"] if x["success"])
    acc["gt_ious"].extend(x["iou"] if x["success"] else 0.0 for x in m50["per_gt"])
    acc["statuses"][str(image_id)] = [x["success"] for x in m50["per_gt"]]
    displayed = [{"box": [float(v) for v in box], "confidence": float(score)} for box, score in zip(boxes, scores)]
    return m50, displayed


def official_metrics(acc: dict[str, Any], annotations: Path, dataset: Any, seconds: float) -> dict[str, Any]:
    coco = COCO(str(annotations))
    result = coco.loadRes(acc["coco_predictions"]) if acc["coco_predictions"] else coco.loadRes([])
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.imgIds = [int(record[0]["id"]) for record in dataset.records]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    person_index = dataset.class_names.index("person")
    precision = evaluator.eval["precision"]
    def class_ap(threshold_index: int | None = None) -> float:
        values = precision[:, :, person_index, 0, -1] if threshold_index is None else precision[threshold_index, :, person_index, 0, -1]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else 0.0
    recalls = evaluator.eval["recall"][:, person_index, 0, -1]
    recalls = recalls[recalls > -1]
    gt = acc["person_gt"]; pred = acc["person_predictions"]; tp = acc["matched50"]
    recall50 = tp / gt if gt else 0.0; precision50 = tp / pred if pred else 0.0
    f1 = 2 * recall50 * precision50 / (recall50 + precision50) if recall50 + precision50 else 0.0
    return {"person_ap": class_ap(), "person_ap50": class_ap(0), "person_ap75": class_ap(5),
            "person_ar100": float(recalls.mean()) if recalls.size else 0.0,
            "person_recall_iou50_conf025": recall50,
            "person_recall_iou75_conf025": acc["matched75"] / gt if gt else 0.0,
            "precision_iou50_conf025": precision50, "f1_iou50_conf025": f1,
            "mean_matched_iou": statistics.fmean(acc["matched_ious"]) if acc["matched_ious"] else 0.0,
            "mean_gt_iou_with_misses": statistics.fmean(acc["gt_ious"]) if acc["gt_ious"] else 0.0,
            "fp_per_image_conf025": (pred - tp) / len(dataset), "person_gt": gt,
            "person_predictions_conf025": pred, "images": len(dataset), "seconds": seconds}


def set_alphas(model: ProposedRGBShapeV3, values: list[float]) -> None:
    with torch.no_grad():
        for fusion, value in zip(model.fusions, values):
            fusion.alpha.fill_(value)


def evaluate_pair(model: ProposedRGBShapeV3, dataset: Any, annotations: Path,
                  condition: str, stored_alphas: list[float], device: torch.device) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=coco_person_collate,
                        num_workers=0, pin_memory=True)
    class_to_category = {v: k for k, v in dataset.category_id_to_class.items()}
    acc = {"v3_rgb_only": empty_accumulator(), "v3_normal": empty_accumulator()}
    images: list[dict[str, Any]] = []
    started = {"v3_rgb_only": 0.0, "v3_normal": 0.0}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["img"].to(device, non_blocking=True)
            set_alphas(model, [0.0, 0.0, 0.0]); torch.cuda.synchronize(); t = time.perf_counter()
            rgb_detections = model.predict(x, confidence=CONFIDENCE, iou=NMS_IOU, classes=None)
            torch.cuda.synchronize(); started["v3_rgb_only"] += time.perf_counter() - t
            set_alphas(model, stored_alphas); torch.cuda.synchronize(); t = time.perf_counter()
            normal_detections = model.predict(x, confidence=CONFIDENCE, iou=NMS_IOU, classes=None)
            torch.cuda.synchronize(); started["v3_normal"] += time.perf_counter() - t
            for index, (rgb_det, normal_det) in enumerate(zip(rgb_detections, normal_detections)):
                image_id = int(batch["image_id"][index]); gt = original_gt(batch, index)
                rgb_restored = inverse_letterbox_xyxy(rgb_det[:, :4].cpu(), batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index])
                normal_restored = inverse_letterbox_xyxy(normal_det[:, :4].cpu(), batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index])
                rgb_match, rgb_display = add_prediction(acc["v3_rgb_only"], rgb_det, rgb_restored, gt, image_id, class_to_category)
                normal_match, normal_display = add_prediction(acc["v3_normal"], normal_det, normal_restored, gt, image_id, class_to_category)
                per_gt = []
                for gt_index, (r, n) in enumerate(zip(rgb_match["per_gt"], normal_match["per_gt"])):
                    transition = ("both_success" if r["success"] and n["success"] else
                                  "shape_helps" if not r["success"] and n["success"] else
                                  "shape_hurts" if r["success"] and not n["success"] else "both_fail")
                    per_gt.append({"gt_index": gt_index, "gt_box": [float(v) for v in gt[gt_index]],
                                   "transition": transition, "rgb": r, "normal": n})
                images.append({"condition": condition, "dataset_index": len(images), "image_id": image_id,
                               "file_name": batch["file_name"][index], "original_size": [float(v) for v in batch["original_size"][index]],
                               "gt": [[float(v) for v in row] for row in gt], "per_gt": per_gt,
                               "rgb_detections": rgb_display, "normal_detections": normal_display})
    set_alphas(model, stored_alphas)
    rgb_metrics = official_metrics(acc["v3_rgb_only"], annotations, dataset, started["v3_rgb_only"])
    normal_metrics = official_metrics(acc["v3_normal"], annotations, dataset, started["v3_normal"])
    rgb_metrics["person_match_status"] = acc["v3_rgb_only"]["statuses"]
    normal_metrics["person_match_status"] = acc["v3_normal"]["statuses"]
    return rgb_metrics, normal_metrics, images


def failure(clean: dict[str, Any], attacked: dict[str, Any]) -> float | None:
    successful = failed = 0
    for image_id, clean_flags in clean["person_match_status"].items():
        attack_flags = attacked["person_match_status"][image_id]
        for c, a in zip(clean_flags, attack_flags):
            successful += int(c); failed += int(c and not a)
    return failed / successful if successful else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def tensor_image(item: dict[str, Any]) -> Image.Image:
    pixels = (item["img"].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(pixels, "RGB")


def original_to_letterbox(box: list[float], item: dict[str, Any]) -> list[float]:
    scale = float(item["letterbox_scale"]); left, top = map(float, item["letterbox_pad"])
    return [box[0]*scale+left, box[1]*scale+top, box[2]*scale+left, box[3]*scale+top]


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: str, label: str) -> None:
    draw.rectangle(box, outline=color, width=3)
    x, y = max(0, int(box[0])), max(0, int(box[1])-18)
    draw.rectangle((x, y, min(639, x+max(75, len(label)*7)), y+18), fill=color)
    draw.text((x+2, y+2), label, fill="white")


def render_case(dataset: Any, record: dict[str, Any], destination: Path, title: str) -> None:
    item = dataset[record["dataset_index"]]
    base = tensor_image(item); panels = []
    for name, key in (("GT", None), ("RGB-ONLY", "rgb_detections"), ("NORMAL v3", "normal_detections")):
        panel = base.copy(); draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 640, 28), fill="black"); draw.text((8, 6), name, fill="white")
        if key is None:
            for box in record["gt"]: draw_box(draw, original_to_letterbox(box, item), "lime", "GT person")
        else:
            for det in record[key]:
                box = torch.tensor(det["box"]).reshape(1, 4); gt = torch.tensor(record["gt"]).reshape(-1, 4)
                overlap = float(box_iou(box, gt).max()) if len(gt) else 0.0
                draw_box(draw, original_to_letterbox(det["box"], item), "lime" if overlap >= .5 else "red",
                         f"person {det['confidence']:.2f} IoU {overlap:.2f}")
        panels.append(panel)
    canvas = Image.new("RGB", (1920, 690), "white"); draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1920, 50), fill="#202020")
    draw.text((10, 16), f"{record['condition'].upper()} | COCO {record['image_id']} | {title}", fill="white")
    for index, panel in enumerate(panels): canvas.paste(panel, (index*640, 50))
    destination.parent.mkdir(parents=True, exist_ok=True); canvas.save(destination, quality=95)


def select_images(records: list[dict[str, Any]], transition: str, quotas: dict[str, int]) -> list[dict[str, Any]]:
    chosen = []
    for condition, quota in quotas.items():
        candidates = [r for r in records if r["condition"] == condition and any(g["transition"] == transition for g in r["per_gt"])]
        candidates.sort(key=lambda r: sum(g["transition"] == transition for g in r["per_gt"]), reverse=True)
        chosen.extend(candidates[:quota])
    return chosen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = config["validation"]["clean"]; images = resolve(val["images"]); annotations = resolve(val["annotations"])
    payload = json.loads(annotations.read_text(encoding="utf-8")); categories = tuple(sorted(int(x["id"]) for x in payload["categories"]))
    common = dict(images_dir=images, annotations_file=annotations, image_size=IMAGE_SIZE, max_images=None,
                  target_policy="person", required_category_ids=(), category_ids=categories, seed=7,
                  position_jitter=.015, placement_mode="torso", torso_relative_y=.38,
                  patch_scale_min=.95, patch_scale_max=1.05, patch_rotation_degrees=3.,
                  patch_brightness_jitter=.03, patch_perspective_jitter=.005)
    patch_dirs = {"seen": (ROOT / "runs" / "v5_diagnostic" / "patch_V5-C.png",),
                  "unseen": (ROOT / "artifacts" / "adversarial_patches" / "unseen",)}
    datasets = {mode: Coco80DetectionDataset(CocoPersonConfig(**common, patch_mode=mode,
                patch_probability=0.0 if mode == "clean" else 1.0,
                patch_dirs=() if mode == "clean" else patch_dirs[mode])) for mode in ("clean", "seen", "unseen")}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required for the full ablation")
    model = ProposedRGBShapeV3(ProposedV3Config(weights=str(ROOT / "yolo11n.pt")))
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False); model.load_state_dict(state["model"], strict=True)
    model.to(device).eval(); stored_alphas = [float(f.alpha.detach()) for f in model.fusions]
    metrics = {"v3_rgb_only": {}, "v3_normal": {}}; records: list[dict[str, Any]] = []
    for condition in ("clean", "seen", "unseen"):
        rgb, normal, image_records = evaluate_pair(model, datasets[condition], annotations, condition, stored_alphas, device)
        metrics["v3_rgb_only"][condition] = rgb; metrics["v3_normal"][condition] = normal; records.extend(image_records)
    for mode in metrics:
        for condition in ("seen", "unseen"):
            metrics[mode][condition]["failure_rate"] = failure(metrics[mode]["clean"], metrics[mode][condition])
        metrics[mode]["clean"]["failure_rate"] = None
        for condition in metrics[mode]: metrics[mode][condition].pop("person_match_status", None)
    (OUT / "rgb_only_metrics.json").write_text(json.dumps(metrics["v3_rgb_only"], indent=2), encoding="utf-8")
    (OUT / "normal_v3_metrics.json").write_text(json.dumps(metrics["v3_normal"], indent=2), encoding="utf-8")
    keys = ("person_ap", "person_ap50", "person_ap75", "person_ar100", "person_recall_iou50_conf025",
            "person_recall_iou75_conf025", "precision_iou50_conf025", "f1_iou50_conf025",
            "mean_matched_iou", "mean_gt_iou_with_misses", "fp_per_image_conf025", "failure_rate")
    comparison = []
    for mode in ("v3_rgb_only", "v3_normal"):
        for condition in ("clean", "seen", "unseen"):
            comparison.append({"mode": mode, "condition": condition, **{k: metrics[mode][condition].get(k) for k in keys}})
    write_csv(OUT / "mode_comparison.csv", comparison)
    deltas = [{"condition": c, **{f"{k}_gain": (metrics["v3_normal"][c].get(k) - metrics["v3_rgb_only"][c].get(k)) if metrics["v3_normal"][c].get(k) is not None and metrics["v3_rgb_only"][c].get(k) is not None else None for k in keys}} for c in ("clean", "seen", "unseen")]
    write_csv(OUT / "mode_differences.csv", deltas)

    transitions = []; iou_rows = []; confidence_rows = []
    for condition in ("clean", "seen", "unseen"):
        gt_rows = [g for r in records if r["condition"] == condition for g in r["per_gt"]]
        counts = {name: sum(g["transition"] == name for g in gt_rows) for name in ("both_success", "shape_helps", "shape_hurts", "both_fail")}
        transitions.append({"condition": condition, "total_gt": len(gt_rows), **counts,
                            "both_success_rate": counts["both_success"]/len(gt_rows),
                            "shape_helps_rate": counts["shape_helps"]/len(gt_rows),
                            "shape_hurts_rate": counts["shape_hurts"]/len(gt_rows),
                            "both_fail_rate": counts["both_fail"]/len(gt_rows),
                            "net_shape_benefit": counts["shape_helps"]-counts["shape_hurts"],
                            "net_benefit_rate": (counts["shape_helps"]-counts["shape_hurts"])/len(gt_rows)})
        both = [g for g in gt_rows if g["transition"] == "both_success"]
        iou_delta = [g["normal"]["iou"]-g["rgb"]["iou"] for g in both]
        conf_delta = [g["normal"]["confidence"]-g["rgb"]["confidence"] for g in both]
        iou_rows.append({"condition": condition, "both_success_gt": len(both), "mean_iou_gain": statistics.fmean(iou_delta) if iou_delta else 0,
                         "median_iou_gain": statistics.median(iou_delta) if iou_delta else 0,
                         "iou_improved_count": sum(x>1e-6 for x in iou_delta), "iou_worsened_count": sum(x < -1e-6 for x in iou_delta),
                         "iou_unchanged_count": sum(abs(x)<=1e-6 for x in iou_delta)})
        confidence_rows.append({"condition": condition, "both_success_gt": len(both), "mean_confidence_change": statistics.fmean(conf_delta) if conf_delta else 0,
                                "median_confidence_change": statistics.median(conf_delta) if conf_delta else 0,
                                "confidence_increased_count": sum(x>1e-6 for x in conf_delta), "confidence_decreased_count": sum(x < -1e-6 for x in conf_delta),
                                "confidence_unchanged_count": sum(abs(x)<=1e-6 for x in conf_delta)})
    write_csv(OUT / "per_gt_transition_counts.csv", transitions); write_csv(OUT / "iou_change_analysis.csv", iou_rows); write_csv(OUT / "confidence_change_analysis.csv", confidence_rows)

    helps = select_images(records, "shape_helps", {"clean": 5, "seen": 5, "unseen": 10})
    hurts = select_images(records, "shape_hurts", {"clean": 5, "seen": 5, "unseen": 10})
    help_rows = []; hurt_rows = []
    for label, selected, folder, rows in (("Shape Helps", helps, "shape_helps", help_rows), ("Shape Hurts", hurts, "shape_hurts", hurt_rows)):
        for index, record in enumerate(selected, 1):
            name = f"{folder}_{index:03d}.jpg"; render_case(datasets[record["condition"]], record, OUT/"shape_ablation_visuals"/folder/name, label)
            rows.append({"file": name, "condition": record["condition"], "image_id": record["image_id"], "gt_count": len(record["gt"]),
                         "matching_gt_indices": ";".join(str(g["gt_index"]) for g in record["per_gt"] if g["transition"] == folder)})
    write_csv(OUT / "shape_helps_cases.csv", help_rows); write_csv(OUT / "shape_hurts_cases.csv", hurt_rows)

    localization_candidates = []
    for record in records:
        changes = [g["normal"]["iou"]-g["rgb"]["iou"] for g in record["per_gt"] if g["transition"] == "both_success"]
        if changes: localization_candidates.append((max(changes, key=abs), record))
    positive = sorted((x for x in localization_candidates if x[0] > 0), key=lambda x: x[0], reverse=True)[:5]
    negative = sorted((x for x in localization_candidates if x[0] < 0), key=lambda x: x[0])[:5]
    loc_rows = []
    for index, (change, record) in enumerate(positive+negative, 1):
        name=f"localization_{index:03d}.jpg"; render_case(datasets[record["condition"]], record, OUT/"shape_ablation_visuals"/"localization_comparison"/name, f"IoU delta {change:+.3f}")
        loc_rows.append({"file":name,"condition":record["condition"],"image_id":record["image_id"],"largest_iou_delta":change})
    write_csv(OUT / "localization_cases.csv", loc_rows)

    nontrivial_hurts = any(row["shape_hurts_rate"] >= .005 for row in transitions)
    all_net_positive = all(row["net_shape_benefit"] > 0 for row in transitions)
    if all_net_positive and not nontrivial_hurts and all(row["person_ap_gain"] >= 0 for row in deltas): verdict = "FIXED ALPHA SUFFICIENT"
    elif all_net_positive: verdict = "ADAPTIVE ALPHA POTENTIALLY USEFUL"
    else: verdict = "FIXED ALPHA LIMITATION CONFIRMED"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = ["# V3 RGB-only vs Shape Fusion Ablation", "", f"- Checkpoint: `{CHECKPOINT}`", f"- Git commit: `{commit}`", f"- Stored alpha: alpha3={stored_alphas[0]:.6f}, alpha4={stored_alphas[1]:.6f}, alpha5={stored_alphas[2]:.6f}", "- RGB-only changes only the in-memory inference contribution to zero; the checkpoint was not modified.", "", "## Overall metrics", "", "| Mode | Condition | AP | AP50 | AP75 | AR100 | R50 | R75 | Precision | F1 | mIoU | GT-IoU | FP/img | Failure |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison:
        vals=[row["mode"],row["condition"]]+[("" if row[k] is None else f"{row[k]:.4f}") for k in keys]
        report.append("| " + " | ".join(vals) + " |")
    report += ["", "## Per-GT transitions", "", "| Condition | Both Success | Shape Helps | Shape Hurts | Both Fail | Net Benefit | Net Rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in transitions: report.append(f"| {row['condition']} | {row['both_success']} | {row['shape_helps']} | {row['shape_hurts']} | {row['both_fail']} | {row['net_shape_benefit']} | {row['net_benefit_rate']:.4%} |")
    report += ["", "## Verdict", "", verdict, "", "See the CSV/JSON files and `shape_ablation_visuals` for complete quantitative and qualitative evidence."]
    (OUT/"ABLATION_REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    (OUT/"run_metadata.json").write_text(json.dumps({"git_commit":commit,"checkpoint":str(CHECKPOINT),"stored_alphas":stored_alphas,"device":torch.cuda.get_device_name(0),"conditions":["clean","seen","unseen"],"confidence":CONFIDENCE,"person_confidence":PERSON_CONFIDENCE,"nms_iou":NMS_IOU,"verdict":verdict},indent=2),encoding="utf-8")
    print(json.dumps({"output":str(OUT),"stored_alphas":stored_alphas,"transitions":transitions,"verdict":verdict},indent=2))


if __name__ == "__main__": main()
