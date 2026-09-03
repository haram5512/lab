"""Short training-step throughput benchmark for the clean Baseline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baseline import BaselineDetector
from .dataset import CocoPersonConfig, CocoPersonPatchDataset, coco_person_collate


def gpu_sample() -> tuple[float | None, float | None]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        memory = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024 ** 2)
        return util, memory
    except Exception:
        return None, None


def run_case(root: Path, weights: str, batch_size: int, workers: int, pin: bool, persistent: bool, steps: int, warmup: int) -> dict[str, object]:
    dataset = CocoPersonPatchDataset(CocoPersonConfig(root / "train2017", root / "annotations/instances_train2017.json", image_size=640, patch_mode="clean", max_images=100))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=coco_person_collate, num_workers=workers, pin_memory=pin, persistent_workers=persistent and workers > 0)
    model = BaselineDetector(weights).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    iterator = iter(loader)
    timings: list[float] = []
    utils: list[float] = []
    memory_samples: list[float] = []
    try:
        for step in range(warmup + steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            model_batch = {key: value.cuda(non_blocking=pin) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            predictions = model(model_batch["img"])
            loss_vector, _ = model.loss(model_batch, predictions)
            loss = loss_vector.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if step >= warmup:
                timings.append(elapsed)
                util, memory = gpu_sample()
                if util is not None:
                    utils.append(util)
                if memory is not None:
                    memory_samples.append(memory)
        mean_step = sum(timings) / len(timings)
        image_rate = batch_size / mean_step
        epoch_seconds = math.ceil(118287 / batch_size) * mean_step
        return {"batch": batch_size, "workers": workers, "pin_memory": pin, "persistent_workers": persistent and workers > 0, "step_seconds": mean_step, "images_sec": image_rate, "epoch_minutes": epoch_seconds / 60, "gpu_util_avg": (sum(utils) / len(utils)) if utils else None, "gpu_mem_mib": max(memory_samples) if memory_samples else round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1), "cuda_oom": False, "loss_last": float(loss.detach())}
    except torch.cuda.OutOfMemoryError:
        return {"batch": batch_size, "workers": workers, "pin_memory": pin, "persistent_workers": persistent and workers > 0, "cuda_oom": True}
    finally:
        del model, optimizer, loader, dataset
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--batches", default="2,4,8,16")
    parser.add_argument("--workers", default="0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline_benchmark.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "datasets/coco"
    results = []
    for batch_size in (int(item) for item in args.batches.split(",")):
        results.append(run_case(root, args.weights, batch_size, int(args.workers), int(args.workers) > 0, int(args.workers) > 0, args.steps, args.warmup))
        print(json.dumps(results[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
