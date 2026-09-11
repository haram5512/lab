"""Fair batch-1 latency benchmark for Sobel, Canny, and Laplacian v3 checkpoints."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from .dataset import Coco80DetectionDataset, CocoPersonConfig
from .models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "main_80class"
OUT = RUNS / "v3_edge_operator_ablation" / "latency_benchmark.json"
CHECKPOINTS = {
    "sobel": RUNS / "proposed_rgb_shape_v3_full" / "best_seen_map.pt",
    "canny": RUNS / "proposed_v3_canny_30ep" / "best_seen_map.pt",
    "laplacian": RUNS / "proposed_v3_laplacian_30ep" / "best_seen_map.pt",
}


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "median_ms": median,
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
        "fps_from_median": 1000.0 / median,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled latency benchmark")
    config = yaml.safe_load((ROOT / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = config["validation"]["clean"]
    images = (ROOT / val["images"]).resolve()
    annotations = (ROOT / val["annotations"]).resolve()
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    categories = tuple(sorted(int(item["id"]) for item in payload["categories"]))
    dataset = Coco80DetectionDataset(CocoPersonConfig(
        images_dir=images, annotations_file=annotations, image_size=640, max_images=140,
        patch_mode="clean", patch_probability=0.0, patch_dirs=(), category_ids=categories,
        required_category_ids=(), seed=7,
    ))
    cached = [dataset[index]["img"].unsqueeze(0) for index in range(120)]
    device = torch.device("cuda:0")
    results: dict[str, object] = {}
    for operator, checkpoint in CHECKPOINTS.items():
        model = ProposedRGBShapeV3(ProposedV3Config(
            weights=str(ROOT / "yolo11n.pt"), edge_operator=operator
        ))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        with torch.inference_mode():
            for index in range(20):
                x = cached[index].to(device)
                model.predict(x, confidence=.25, iou=.7, classes=[0])
            torch.cuda.synchronize()

            edge_times = []
            model_times = []
            for x_cpu in cached[20:120]:
                x = x_cpu.to(device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                model.shape_extractor(x)
                torch.cuda.synchronize()
                edge_times.append((time.perf_counter() - started) * 1000.0)
                started = time.perf_counter()
                model.predict(x, confidence=.25, iou=.7, classes=[0])
                torch.cuda.synchronize()
                model_times.append((time.perf_counter() - started) * 1000.0)

            decode_times = []
            end_to_end_times = []
            for index in range(20, 120):
                started = time.perf_counter()
                item = dataset[index]
                prepared = time.perf_counter()
                model.predict(item["img"].unsqueeze(0).to(device), confidence=.25, iou=.7, classes=[0])
                torch.cuda.synchronize()
                finished = time.perf_counter()
                decode_times.append((prepared - started) * 1000.0)
                end_to_end_times.append((finished - started) * 1000.0)
        results[operator] = {
            "checkpoint": str(checkpoint), "checkpoint_epoch": int(state["epoch"]),
            "edge_preprocessing": model.edge_preprocessing,
            "edge_preprocessing_gpu": summary(edge_times),
            "model_including_edge_plus_nms": summary(model_times),
            "cpu_decode_letterbox": summary(decode_times),
            "end_to_end_decode_letterbox_h2d_model_nms": summary(end_to_end_times),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        }
        del model
        torch.cuda.empty_cache()
    report = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "input": "100 real COCO val2017 images, batch=1, letterbox 640x640",
        "precision": "FP32", "warmup": 20, "repetitions": 100,
        "confidence": .25, "nms_iou": .7, "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
