"""Short clean-only Baseline training and full-training entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .baseline import BaselineDetector
from .dataset import CocoPersonConfig, CocoPersonPatchDataset, coco_person_collate
from .metrics import evaluate_coco_person


def make_loader(project_root: Path, config_path: Path, split: str, max_images: int | None, image_size: int, batch_size: int, num_workers: int = 0, pin_memory: bool = False, persistent_workers: bool = False) -> DataLoader:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_config = config[split]["clean"] if split == "validation" else config[split]
    images = Path(split_config["images"])
    annotations = Path(split_config["annotations"])
    if not images.is_absolute():
        images = project_root / images
    if not annotations.is_absolute():
        annotations = project_root / annotations
    dataset = CocoPersonPatchDataset(
        CocoPersonConfig(images, annotations, image_size=image_size, patch_mode="clean", max_images=max_images)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=coco_person_collate, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers and num_workers > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("dataset_config.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--split", choices=("train", "validation.clean"), default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=100, help="0 means all person images in the split.")
    parser.add_argument("--val-max-images", type=int, default=0, help="0 means all person images in val2017.")
    parser.add_argument("--val-every", type=int, default=5, help="Run validation every N epochs; use 1 for every epoch.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline_clean_smoke"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    split = "train" if args.split == "train" else "validation"
    max_images = None if args.max_images <= 0 else args.max_images
    loader = make_loader(project_root, config_path, split, max_images, args.image_size, args.batch_size, args.num_workers, args.pin_memory, args.persistent_workers)
    val_max_images = None if args.val_max_images <= 0 else args.val_max_images
    val_loader = make_loader(project_root, config_path, "validation", val_max_images, args.image_size, args.batch_size, args.num_workers, args.pin_memory, args.persistent_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineDetector(args.weights).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_map = float("-inf")
    if args.val_every < 1:
        raise ValueError("val_every must be at least 1.")

    for epoch in range(args.epochs):
        losses = []
        for batch in loader:
            images = batch["img"].to(device)
            model_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss_vector, components = model.loss(model_batch, predictions)
            loss = loss_vector.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = sum(losses) / len(losses)
        should_validate = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        val_mean_loss = None
        val_map = None
        val_ap50 = None
        val_ap75 = None
        validation_duration = None
        map_duration = None
        validation_batches = 0
        if should_validate:
            evaluation = evaluate_coco_person(model, val_loader, device, args.image_size)
            val_mean_loss = evaluation["val_loss"]
            val_map = evaluation["map"]
            val_ap50 = evaluation["ap50"]
            val_ap75 = evaluation["ap75"]
            validation_duration = evaluation["val_seconds"]
            map_duration = evaluation["map_seconds"]
            validation_batches = len(val_loader)
            print(f"epoch={epoch + 1} train_loss={mean_loss:.6f} val_loss={val_mean_loss:.6f} map={val_map:.4f} ap50={val_ap50:.4f} ap75={val_ap75:.4f} val_seconds={validation_duration:.1f} map_seconds={map_duration:.1f} device={device}")
        else:
            print(f"epoch={epoch + 1} train_loss={mean_loss:.6f} validation=skipped train_batches={len(losses)} device={device}")
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps({"epoch": epoch + 1, "train_loss": mean_loss, "val_loss": val_mean_loss, "map": val_map, "ap50": val_ap50, "ap75": val_ap75, "train_batches": len(losses), "val_batches": validation_batches, "val_seconds": validation_duration, "map_seconds": map_duration, "device": str(device)}) + "\n")
        state = {"model": model.state_dict(), "epoch": epoch + 1, "train_loss": mean_loss, "val_loss": val_mean_loss, "map": val_map, "ap50": val_ap50, "ap75": val_ap75}
        torch.save(state, args.output / "last.pt")
        if val_map is not None and val_map > best_map:
            best_map = val_map
            torch.save(state, args.output / "best.pt")
            torch.save(state, args.output / "best_map.pt")
        if val_mean_loss is not None and val_mean_loss < best_loss:
            best_loss = val_mean_loss
            torch.save(state, args.output / "best_loss.pt")


if __name__ == "__main__":
    main()
