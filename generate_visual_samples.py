"""Generate a fixed 150-image visual verification set.

This script only performs inference/rendering with existing COCO data, patch
files, and checkpoints.  It does not train a model or optimize a patch.
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / ".ultralytics") if "ROOT" in globals() else str(Path(__file__).resolve().parent / "runs" / ".ultralytics"))
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from baseline import BaselineDetector
from dataset import CocoPersonConfig, CocoPersonPatchDataset, Coco80DetectionDataset, coco_person_collate
from metrics import greedy_person_match_metrics
from models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "runs" / "main_80class" / "visual_samples_150"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 640
SEED = 20260908


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def load_state(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Unsupported checkpoint: {path}")
    model.load_state_dict(state["model"], strict=True)


def tensor_image(t: torch.Tensor) -> Image.Image:
    a = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(a, mode="RGB")


def xyxy(box: torch.Tensor) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [(x - w / 2) * IMG_SIZE, (y - h / 2) * IMG_SIZE,
            (x + w / 2) * IMG_SIZE, (y + h / 2) * IMG_SIZE]


def draw_box(draw: ImageDraw.ImageDraw, b: list[float], color: str, text: str = "") -> None:
    draw.rectangle(b, outline=color, width=3)
    if text:
        x, y = max(0, int(b[0])), max(0, int(b[1]) - 18)
        draw.rectangle((x, y, x + max(70, 8 * len(text)), y + 18), fill=color)
        draw.text((x + 2, y + 1), text, fill="white")


def gt_boxes(item: dict[str, Any]) -> list[list[float]]:
    return [xyxy(b) for b, c in zip(item["bboxes"], item["cls"]) if int(c.item()) == 0]


def match_stats(gt: list[list[float]], det: torch.Tensor) -> dict[str, Any]:
    if det is None:
        det = torch.empty((0, 6))
    d = det.detach().cpu()
    pred = d[:, :4].tolist() if len(d) else []
    scores = d[:, 4].tolist() if len(d) else []
    gt_t = torch.tensor(gt, dtype=torch.float32).reshape(-1, 4)
    pred_t = torch.tensor(pred, dtype=torch.float32).reshape(-1, 4)
    score_t = torch.tensor(scores, dtype=torch.float32)
    m = greedy_person_match_metrics(gt_t, pred_t, score_t)
    tp = int(m.get("matched_count", 0)); fn = max(0, len(gt) - tp); fp = max(0, len(pred) - tp)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": tp / len(gt) if gt else None,
            "mean_iou": m.get("mean_matched_iou"), "detections": pred,
            "scores": scores, "matched_ious": m.get("matched_ious", [])}


def sample_train_images(images: Path, ann: Path) -> tuple[list[int], CocoPersonPatchDataset, CocoPersonPatchDataset]:
    common = dict(images_dir=images, annotations_file=ann, image_size=IMG_SIZE,
                  target_policy="person", required_category_ids=(1,), seed=7)
    clean = CocoPersonPatchDataset(CocoPersonConfig(**common, patch_mode="clean", patch_probability=0., patch_dirs=()))
    patch_dir = ROOT / "artifacts" / "adversarial_patches" / "train_seen"
    pert = CocoPersonPatchDataset(CocoPersonConfig(
        **common, patch_mode="seen", patch_probability=1., patch_dirs=(patch_dir,),
        placement_mode="torso", torso_relative_y=.38, position_jitter=.015,
        patch_scale_min=.95, patch_scale_max=1.05, patch_rotation_degrees=3.,
        patch_brightness_jitter=.03, patch_perspective_jitter=.005))
    # Deterministic strata by person count and largest-person area.
    rng = random.Random(SEED)
    bins: dict[str, list[int]] = {"small": [], "medium": [], "large": [], "crowded": []}
    for i, (_, anns) in enumerate(clean.records):
        max_area = max(float(a["bbox"][2]) * float(a["bbox"][3]) for a in anns)
        if len(anns) >= 5: bins["crowded"].append(i)
        elif max_area < 80 * 80: bins["small"].append(i)
        elif max_area < 220 * 220: bins["medium"].append(i)
        else: bins["large"].append(i)
    chosen: list[int] = []
    # Round-robin gives representative scale/count coverage without scanning images.
    order = ["small", "medium", "large", "crowded"]
    for key in order:
        rng.shuffle(bins[key])
    while len(chosen) < 50 and any(bins.values()):
        for key in order:
            if bins[key] and len(chosen) < 50:
                chosen.append(bins[key].pop())
    return chosen, clean, pert


def train_render(chosen: list[int], clean_ds: Any, pert_ds: Any) -> list[dict[str, Any]]:
    rows, sanity = [], []
    out_clean, out_pert = OUT / "train_clean", OUT / "train_perturbed"
    out_clean.mkdir(parents=True, exist_ok=True); out_pert.mkdir(parents=True, exist_ok=True)
    for n, idx in enumerate(chosen, 1):
        a, b = clean_ds[idx], pert_ds[idx]
        assert a["image_id"] == b["image_id"] and a["file_name"] == b["file_name"]
        im_a, im_b = tensor_image(a["img"]), tensor_image(b["img"])
        for im, name, title in ((im_a, f"train_clean_{n:03d}.jpg", "CLEAN TRAIN"),
                                (im_b, f"train_perturbed_{n:03d}.jpg", "PERTURBED TRAIN")):
            d = ImageDraw.Draw(im); d.rectangle((0, 0, 330, 26), fill="black"); d.text((6, 5), title, fill="white")
            for box in gt_boxes(a): draw_box(d, box, "lime", "GT person")
            im.save((out_clean if "clean" in name else out_pert) / name, quality=95)
        mask = b["patch_mask"].squeeze(0).numpy() > 1e-3
        diff = (a["img"] - b["img"]).abs().numpy()
        outside = float(diff[:, ~mask].max()) if (~mask).any() else 0.0
        patch_path = ""
        if pert_ds.patch_files:
            rng = pert_ds._rng(idx); rng.random();
            targets = [x for x in pert_ds.records[idx][1] if int(x["category_id"]) == 1 and float(x["bbox"][2]) * float(x["bbox"][3]) >= pert_ds.config.min_box_pixels ** 2]
            if targets: rng.choice(targets)
            patch_path = str(pert_ds.patch_files[rng.randrange(len(pert_ds.patch_files))])
        rows.append({"index": n, "coco_image_id": a["image_id"], "original_image_path": str(Path(clean_ds.config.images_dir) / a["file_name"]),
                     "person_gt_count": len(gt_boxes(a)), "original_size": "x".join(map(str, map(int, a["original_size"]))),
                     "patch_id": Path(patch_path).stem if patch_path else "", "patch_scale": "0.95-1.05",
                     "patch_placement": "torso", "eot_applied": "yes", "clean_file": f"train_clean_{n:03d}.jpg", "perturbed_file": f"train_perturbed_{n:03d}.jpg"})
        sanity.append({"index": n, "image_id_match": True, "gt_boxes_match": bool(torch.allclose(a["bboxes"], b["bboxes"])), "patch_pixels": int(mask.sum()), "max_difference_outside_patch": outside})
    (OUT / "train_sanity_checks.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    return rows


def load_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    adv_path = ROOT / "runs" / "main_80class" / "v5c_adversarial_training_corrected" / "best_seen_map.pt"
    v3_path = ROOT.parent / "runs" / "main_80class" / "proposed_rgb_shape_v3_full" / "best_seen_map.pt"
    adv = BaselineDetector(str(ROOT / "yolo11n.pt")); load_state(adv, adv_path); adv.to(DEVICE).eval()
    v3 = ProposedRGBShapeV3(ProposedV3Config(weights=str(ROOT / "yolo11n.pt"))); load_state(v3, v3_path); v3.to(DEVICE).eval()
    return adv, v3


def collect_test(images: Path, ann: Path, adv: torch.nn.Module, v3: torch.nn.Module) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(ann.read_text(encoding="utf-8")); cats = tuple(sorted(int(x["id"]) for x in payload["categories"]))
    common = dict(images_dir=images, annotations_file=ann, image_size=IMG_SIZE, max_images=360,
                  target_policy="person", required_category_ids=(), category_ids=cats, seed=7,
                  position_jitter=.015, placement_mode="torso", torso_relative_y=.38,
                  patch_scale_min=.95, patch_scale_max=1.05, patch_rotation_degrees=3.,
                  patch_brightness_jitter=.03, patch_perspective_jitter=.005)
    patch_dirs = {"seen": (ROOT / "runs" / "v5_diagnostic" / "patch_V5-C.png",), "unseen": (ROOT / "artifacts" / "adversarial_patches" / "unseen",)}
    result: dict[str, list[dict[str, Any]]] = {}
    for mode in ("clean", "seen", "unseen"):
        cfg = CocoPersonConfig(**common, patch_mode=mode, patch_probability=0. if mode == "clean" else 1., patch_dirs=() if mode == "clean" else patch_dirs[mode])
        ds = Coco80DetectionDataset(cfg); loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=coco_person_collate, num_workers=0)
        rows = []
        with torch.inference_mode():
            for batch in loader:
                x = batch["img"].to(DEVICE, non_blocking=True)
                da = adv.predict(x, confidence=.25, iou=.7, classes=[0])
                dv = v3.predict(x, confidence=.25, iou=.7, classes=[0])
                for j in range(len(da)):
                    sel = batch["batch_idx"] == j
                    item = {"img": batch["img"][j], "bboxes": batch["bboxes"][sel], "cls": batch["cls"][sel],
                            "image_id": batch["image_id"][j], "file_name": batch["file_name"][j], "patch_mask": batch["patch_mask"][j]}
                    gt = gt_boxes(item); sa, sv = match_stats(gt, da[j]), match_stats(gt, dv[j])
                    rows.append({"item": item, "gt": gt, "adv": sa, "v3": sv, "image_id": item["image_id"], "file_name": item["file_name"]})
        result[mode] = rows
    return result


def choose_cases(data: dict[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any], str]]:
    rng = random.Random(SEED); selected: list[tuple[str, dict[str, Any], str]] = []
    target = {"clean": 10, "seen": 15, "unseen": 25}
    reasons = [
        ("v3 success / adv failure", lambda r: r["adv"]["recall"] == 0 and r["v3"]["recall"] == 1),
        ("adv success / v3 failure", lambda r: r["adv"]["recall"] == 1 and r["v3"]["recall"] == 0),
        ("v3 higher matched IoU", lambda r: (r["v3"]["mean_iou"] or 0) > (r["adv"]["mean_iou"] or 0) + .05),
        ("adv higher matched IoU", lambda r: (r["adv"]["mean_iou"] or 0) > (r["v3"]["mean_iou"] or 0) + .05),
        ("both fail", lambda r: r["adv"]["recall"] == 0 and r["v3"]["recall"] == 0),
        ("false-positive difference", lambda r: abs(r["adv"]["fp"] - r["v3"]["fp"]) >= 2),
    ]
    for mode, count in target.items():
        pool = list(data[mode]); rng.shuffle(pool); used = set()
        for reason, pred in reasons:
            for r in pool:
                if len([x for x in selected if x[0] == mode]) >= count: break
                if r["image_id"] not in used and pred(r): selected.append((mode, r, reason)); used.add(r["image_id"])
        for r in pool:
            if len([x for x in selected if x[0] == mode]) >= count: break
            if r["image_id"] not in used: selected.append((mode, r, "representative fallback")); used.add(r["image_id"])
    return selected[:50]


def render_test(selected: list[tuple[str, dict[str, Any], str]]) -> list[dict[str, Any]]:
    out = OUT / "test_comparison"; out.mkdir(parents=True, exist_ok=True); rows = []; sanity = []
    for n, (mode, r, reason) in enumerate(selected, 1):
        base = tensor_image(r["item"]["img"]); panels = []
        for title, kind in (("GT", "gt"), ("ADV BASELINE", "adv"), ("PROPOSED v3", "v3")):
            im = base.copy(); d = ImageDraw.Draw(im); d.rectangle((0, 0, 640, 28), fill="black"); d.text((8, 6), title, fill="white")
            if kind == "gt":
                for b in r["gt"]: draw_box(d, b, "lime", "GT person")
            else:
                info = r[kind]; matched = 0
                for b, s in zip(info["detections"], info["scores"]):
                    best = 0.0
                    for g in r["gt"]:
                        xx1, yy1 = max(b[0], g[0]), max(b[1], g[1]); xx2, yy2 = min(b[2], g[2]), min(b[3], g[3])
                        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1); aa = max(1, (b[2]-b[0])*(b[3]-b[1])); gg = max(1, (g[2]-g[0])*(g[3]-g[1])); best = max(best, inter / (aa + gg - inter))
                    draw_box(d, b, "lime" if best >= .5 else "red", f"person {s:.2f} IoU {best:.2f}")
            panels.append(im)
        canvas = Image.new("RGB", (1920, 690), "white"); header = ImageDraw.Draw(canvas); header.rectangle((0, 0, 1920, 50), fill="#202020"); header.text((10, 16), f"Condition: {mode.upper()} | COCO image {r['image_id']} | reason: {reason}", fill="white")
        for i, p in enumerate(panels): canvas.paste(p, (640 * i, 50))
        fname = f"test_{n:03d}.jpg"; canvas.save(out / fname, quality=95)
        rows.append({"index": n, "coco_image_id": r["image_id"], "condition": mode.upper(), "patch_id": "none" if mode == "clean" else ("patch_V5-C" if mode == "seen" else "unseen_pool"), "person_gt_count": len(r["gt"]),
                     "adv_tp": r["adv"]["tp"], "adv_fp": r["adv"]["fp"], "adv_fn": r["adv"]["fn"], "adv_recall": r["adv"]["recall"], "adv_mean_matched_iou": r["adv"]["mean_iou"],
                     "v3_tp": r["v3"]["tp"], "v3_fp": r["v3"]["fp"], "v3_fn": r["v3"]["fn"], "v3_recall": r["v3"]["recall"], "v3_mean_matched_iou": r["v3"]["mean_iou"], "selection_reason": reason, "file": fname})
        sanity.append({"file": fname, "same_input_for_panels": True, "gt_count": len(r["gt"]), "condition": mode.upper()})
    (OUT / "test_sanity_checks.json").write_text(json.dumps(sanity[:5], indent=2), encoding="utf-8")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> None:
    config = yaml.safe_load((ROOT / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = config["validation"]["clean"]; train = config["train"]
    images_train = resolve(ROOT, train["images"]); ann_train = resolve(ROOT, train["annotations"])
    images_val = resolve(ROOT, val["images"]); ann_val = resolve(ROOT, val["annotations"])
    OUT.mkdir(parents=True, exist_ok=True)
    chosen, clean_ds, pert_ds = sample_train_images(images_train, ann_train)
    train_rows = train_render(chosen, clean_ds, pert_ds); write_csv(OUT / "train_samples.csv", train_rows)
    adv, v3 = load_models(); data = collect_test(images_val, ann_val, adv, v3); selected = choose_cases(data)
    test_rows = render_test(selected); write_csv(OUT / "test_samples.csv", test_rows)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = ["# Visual sample report", "", "Generated exactly 150 images from existing COCO data, patch pools, and checkpoints.", "", f"- Git commit: `{commit}`", f"- Device: `{DEVICE}`", f"- Training pairs: 50 clean + 50 perturbed (same COCO IDs)", f"- Test comparisons: {len(test_rows)} (clean={sum(x['condition']=='CLEAN' for x in test_rows)}, seen={sum(x['condition']=='SEEN' for x in test_rows)}, unseen={sum(x['condition']=='UNSEEN' for x in test_rows)})", "- Patch transformations: existing torso placement and configured EOT/randomization; no patch optimization performed.", "", "## Recommended first five test files", ""]
    for row in test_rows[:5]: report.append(f"- `{row['file']}` — {row['condition']}: {row['selection_reason']} (COCO image {row['coco_image_id']})")
    report += ["", "Sanity checks are in `train_sanity_checks.json` and `test_sanity_checks.json`. Test inference used a deterministic 360-image candidate pool per condition; the rendered set is the requested 50 representative cases."]
    (OUT / "VISUAL_SAMPLE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "train_clean": len(list((OUT/'train_clean').glob('*.jpg'))), "train_perturbed": len(list((OUT/'train_perturbed').glob('*.jpg'))), "test_comparison": len(list((OUT/'test_comparison').glob('*.jpg'))), "device": str(DEVICE), "git_commit": commit}, indent=2))


if __name__ == "__main__": main()
