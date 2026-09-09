"""Batch-1 real-image latency benchmark for Proposed RGB/Shape v3."""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "runs" / ".ultralytics"))

import torch
import yaml

from dataset import Coco80DetectionDataset, CocoPersonConfig
from models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / "runs" / "main_80class" / "proposed_rgb_shape_v3_full"
CHECKPOINT = RUN / "best_seen_map.pt"
OUTPUT = RUN / "inference_latency_benchmark.json"


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    median = statistics.median(ordered)
    return {
        "median_ms": median,
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": p95,
        "fps_from_median": 1000.0 / median,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cfg = yaml.safe_load((ROOT / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = cfg["validation"]["clean"]
    images = (ROOT / val["images"]).resolve()
    annotations = (ROOT / val["annotations"]).resolve()
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    categories = tuple(sorted(int(x["id"]) for x in payload["categories"]))
    dataset = Coco80DetectionDataset(CocoPersonConfig(
        images_dir=images, annotations_file=annotations, image_size=640,
        max_images=140, patch_mode="clean", patch_probability=0.0,
        patch_dirs=(), category_ids=categories, required_category_ids=(), seed=7,
    ))
    device = torch.device("cuda:0")
    model = ProposedRGBShapeV3(ProposedV3Config(weights=str(ROOT / "yolo11n.pt")))
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    # Cache tensors for a model-only measurement independent of disk decoding.
    cached = [dataset[i]["img"].unsqueeze(0) for i in range(120)]
    with torch.inference_mode():
        for i in range(20):
            x = cached[i].to(device)
            model.predict(x, confidence=0.25, iou=0.7, classes=[0])
        torch.cuda.synchronize()

        inference = []
        for x_cpu in cached[20:120]:
            x = x_cpu.to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            model.predict(x, confidence=0.25, iou=0.7, classes=[0])
            torch.cuda.synchronize()
            inference.append((time.perf_counter() - started) * 1000.0)

        end_to_end = []
        preprocessing = []
        for index in range(20, 120):
            started = time.perf_counter()
            item = dataset[index]
            prepared = time.perf_counter()
            x = item["img"].unsqueeze(0).to(device)
            model.predict(x, confidence=0.25, iou=0.7, classes=[0])
            torch.cuda.synchronize()
            finished = time.perf_counter()
            preprocessing.append((prepared - started) * 1000.0)
            end_to_end.append((finished - started) * 1000.0)

    result = {
        "model": "Proposed RGB/Shape v3",
        "checkpoint": str(CHECKPOINT),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "input": "100 real COCO val2017 images, batch=1, letterbox 640x640",
        "confidence": 0.25,
        "nms_iou": 0.7,
        "warmup": 20,
        "repetitions": 100,
        "precision": "FP32",
        "model_plus_nms": summary(inference),
        "cpu_decode_letterbox": summary(preprocessing),
        "end_to_end_decode_letterbox_h2d_model_nms": summary(end_to_end),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
