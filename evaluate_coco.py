"""Official COCO 80-class evaluation for the pretrained or fine-tuned model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .baseline import BaselineDetector
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .metrics import inverse_letterbox_xyxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("dataset_config.main80.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--attack-mode", choices=("clean", "seen", "unseen"), default="clean")
    parser.add_argument("--patch-root", type=Path, default=Path("artifacts/adversarial_patches"))
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline_pretrained_clean_eval"))
    args = parser.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit("Install pycocotools before official COCO evaluation.") from exc

    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = config["validation"]["clean"]
    images = Path(split["images"])
    annotations = Path(split["annotations"])
    if not images.is_absolute():
        images = project_root / images
    if not annotations.is_absolute():
        annotations = project_root / annotations

    patch_mode = "clean" if args.attack_mode == "clean" else "seen"
    patch_dir = args.patch_root if args.patch_root.is_absolute() else project_root / args.patch_root
    pool_dir = patch_dir / ("train_seen" if args.attack_mode == "seen" else "unseen")
    patch_dirs = () if args.attack_mode == "clean" else (pool_dir,)
    dataset = Coco80DetectionDataset(CocoPersonConfig(images, annotations, image_size=args.image_size, patch_mode=patch_mode, patch_dirs=patch_dirs, patch_probability=1.0 if args.attack_mode != "clean" else 0.0, target_policy="all", max_images=None if args.max_images <= 0 else args.max_images))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=coco_person_collate, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineDetector(args.weights).to(device).eval()
    class_to_category = {class_index: category_id for category_id, class_index in dataset.category_id_to_class.items()}
    predictions: list[dict[str, object]] = []
    attacked_total = 0
    attacked_detected = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            images_tensor = batch["img"].to(device, non_blocking=device.type == "cuda")
            detections = model.predict(images_tensor, confidence=args.confidence, iou=args.iou, classes=None)
            for index, detection in enumerate(detections):
                restored = inverse_letterbox_xyxy(
                    detection[:, :4].cpu(), batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index]
                )
                for box, score, class_index in zip(restored, detection[:, 4].cpu(), detection[:, 5].long().cpu()):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    predictions.append({
                        "image_id": int(batch["image_id"][index]),
                        "category_id": int(class_to_category[int(class_index)]),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    })
                if args.attack_mode != "clean" and batch["patch_target"][index] is not None:
                    target = torch.tensor(batch["patch_target"][index], dtype=torch.float32)
                    target_box = inverse_letterbox_xyxy(
                        torch.tensor([[ (target[0] - target[2] / 2) * args.image_size, (target[1] - target[3] / 2) * args.image_size, (target[0] + target[2] / 2) * args.image_size, (target[1] + target[3] / 2) * args.image_size ]]),
                        batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index],
                    )[0]
                    attacked_total += 1
                    if len(detection):
                        candidate = detection[detection[:, 5].long().cpu() == int(batch["patch_target_class"][index])]
                        if len(candidate):
                            candidate_boxes = inverse_letterbox_xyxy(candidate[:, :4].cpu(), batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index])
                            lt = torch.maximum(candidate_boxes[:, :2], target_box[:2]); rb = torch.minimum(candidate_boxes[:, 2:], target_box[2:]); wh = (rb-lt).clamp(min=0); inter = wh[:,0]*wh[:,1]
                            area_t=(target_box[2]-target_box[0])*(target_box[3]-target_box[1]); area_c=(candidate_boxes[:,2]-candidate_boxes[:,0])*(candidate_boxes[:,3]-candidate_boxes[:,1]);
                            if bool((inter/(area_t+area_c-inter).clamp(min=1e-9) >= 0.5).any()): attacked_detected += 1

    coco = COCO(str(annotations))
    result = coco.loadRes(predictions) if predictions else coco.loadRes([])
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.imgIds = [int(record[0]["id"]) for record in dataset.records]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]
    class_ap: dict[str, float] = {}
    for class_index, name in enumerate(dataset.class_names):
        values = precision[:, :, class_index, 0, -1]
        valid = values[values > -1]
        class_ap[name] = float(valid.mean()) if valid.size else 0.0
    output = {
        "weights": args.weights,
        "attack_mode": args.attack_mode,
        "patch_pool": str(pool_dir.resolve()) if args.attack_mode != "clean" else None,
        "dataset": "COCO 2017 validation, 80 classes",
        "image_size": args.image_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "images": len(dataset),
        "prediction_seconds": time.perf_counter() - started,
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "person_ap": class_ap.get("person"),
        "attacked_object_detection_rate": (attacked_detected / attacked_total) if attacked_total else None,
        "class_ap": class_ap,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "class_ap"}, indent=2))


if __name__ == "__main__":
    main()
