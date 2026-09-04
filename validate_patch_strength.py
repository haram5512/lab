"""Measure patch suppression strength without claiming a research result."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .attacks.patch_losses import target_suppression_loss
from .baseline import BaselineDetector
from .dataset import Coco80DetectionDataset, CocoPersonConfig


def iou(one: torch.Tensor, many: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(one[:2], many[:, :2]); rb = torch.minimum(one[2:], many[:, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[:, 0] * wh[:, 1]
    area_one = (one[2] - one[0]).clamp(min=0) * (one[3] - one[1]).clamp(min=0)
    area_many = (many[:, 2] - many[:, 0]).clamp(min=0) * (many[:, 3] - many[:, 1]).clamp(min=0)
    return inter / (area_one + area_many - inter).clamp(min=1e-9)


def post_nms_score(model: BaselineDetector, image: torch.Tensor, target: torch.Tensor, target_class: int, device: torch.device) -> float:
    detections = model.predict(image.unsqueeze(0).to(device), classes=None)[0].cpu()
    if not len(detections): return 0.0
    boxes = detections[:, :4] / image.shape[-1]
    target_xywh = target.cpu(); cx, cy, width, height = target_xywh
    target_xyxy = torch.tensor([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
    mask = detections[:, 5].long() == int(target_class)
    if not mask.any(): return 0.0
    overlaps = iou(target_xyxy, boxes[mask])
    return float(detections[mask][overlaps >= 0.5, 4].max()) if (overlaps >= 0.5).any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=("train_seen", "unseen"), required=True)
    parser.add_argument("--patch-root", type=Path, default=Path("artifacts/adversarial_patches"))
    parser.add_argument("--config", type=Path, default=Path("dataset_config.main80.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    split = config["validation"]["clean"]
    images, annotations = Path(split["images"]), Path(split["annotations"])
    if not images.is_absolute(): images = root / images
    if not annotations.is_absolute(): annotations = root / annotations
    clean = Coco80DetectionDataset(CocoPersonConfig(images, annotations, image_size=640, max_images=args.max_images))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineDetector(args.weights).to(device).eval()
    pool_dir = args.patch_root if args.patch_root.is_absolute() else root / args.patch_root
    files = sorted((pool_dir / args.pool).glob("patch_*.png"))
    if not files: raise SystemExit(f"No patches found in {pool_dir / args.pool}")
    rows=[]
    for patch_file in files:
        patched = Coco80DetectionDataset(CocoPersonConfig(images, annotations, image_size=640, patch_mode="seen", patch_dirs=(patch_file,), patch_probability=1.0, target_policy="all", max_images=args.max_images))
        before_raw=[]; after_raw=[]; before_nms=[]; after_nms=[]
        for index in range(len(clean)):
            clean_sample=clean[index]; patched_sample=patched[index]
            if patched_sample["patch_target"] is None: continue
            box=torch.tensor([[patched_sample["patch_target"]]],device=device,dtype=torch.float32)
            cls=torch.tensor([[patched_sample["patch_target_class"]]],device=device,dtype=torch.float32)
            with torch.no_grad():
                clean_raw=model(clean_sample["img"].unsqueeze(0).to(device)); patched_raw=model(patched_sample["img"].unsqueeze(0).to(device))
                clean_prediction=clean_raw[0] if isinstance(clean_raw,(tuple,list)) else clean_raw
                patched_prediction=patched_raw[0] if isinstance(patched_raw,(tuple,list)) else patched_raw
                before_raw.append(float(target_suppression_loss(clean_prediction,box,cls,640)))
                after_raw.append(float(target_suppression_loss(patched_prediction,box,cls,640)))
            before_nms.append(post_nms_score(model,clean_sample["img"],torch.tensor(patched_sample["patch_target"]),patched_sample["patch_target_class"],device))
            after_nms.append(post_nms_score(model,patched_sample["img"],torch.tensor(patched_sample["patch_target"]),patched_sample["patch_target_class"],device))
        rows.append({"patch":patch_file.name,"images":len(before_raw),"clean_raw_mean":sum(before_raw)/len(before_raw),"patched_raw_mean":sum(after_raw)/len(after_raw),"raw_score_drop":(sum(before_raw)-sum(after_raw))/len(before_raw),"clean_detection_retention":sum(s>0 for s in before_nms)/len(before_nms),"patched_detection_retention":sum(s>0 for s in after_nms)/len(after_nms),"clean_nms_score_mean":sum(before_nms)/len(before_nms),"patched_nms_score_mean":sum(after_nms)/len(after_nms)})
    output=args.output or root / f"runs/patch_strength_{args.pool}.json"; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps({"pool":args.pool,"images_per_patch":args.max_images,"rows":rows,"note":"Smoke strength diagnostic, not a final research result."},indent=2),encoding="utf-8"); print(json.dumps({"pool":args.pool,"rows":rows},indent=2))


if __name__ == "__main__": main()
