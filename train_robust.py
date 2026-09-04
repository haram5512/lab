"""80-class clean/patch fine-tuning for Baseline or Proposed models."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .baseline import BaselineDetector
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .losses import RobustTrainingLoss
from .model import ModelConfig, ShapeAwareYolo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("dataset_config.main80.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--model", choices=("baseline", "proposed"), default="baseline")
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-probability", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/main_80class/adversarial_training_smoke"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = config["train"]
    images, annotations = Path(train["images"]), Path(train["annotations"])
    if not images.is_absolute(): images = root / images
    if not annotations.is_absolute(): annotations = root / annotations
    patch_dir = args.patch_dir if args.patch_dir.is_absolute() else root / args.patch_dir
    ds_config = CocoPersonConfig(images, annotations, image_size=args.image_size, patch_mode="clean", max_images=None if args.max_images <= 0 else args.max_images, target_policy="all", patch_probability=args.patch_probability)
    clean_ds = Coco80DetectionDataset(ds_config)
    patched_ds = Coco80DetectionDataset(CocoPersonConfig(images, annotations, image_size=args.image_size, patch_mode="seen", patch_dirs=(patch_dir,), max_images=None if args.max_images <= 0 else args.max_images, target_policy="all", patch_probability=1.0))
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coco_person_collate, num_workers=args.num_workers)
    patched_loader = DataLoader(patched_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coco_person_collate, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineDetector(args.weights) if args.model == "baseline" else ShapeAwareYolo(ModelConfig(weights=args.weights, num_classes=None))
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = RobustTrainingLoss(model) if args.model == "proposed" else None
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (args.output / "run_metadata.json").write_text(json.dumps({"git_commit": commit, "model": args.model, "source_checkpoint": args.weights, "dataset": "COCO 2017 (80 classes)", "patch_pool": str(patch_dir.resolve()), "clean_patch_ratio": [0.5, 0.5], "image_size": args.image_size, "batch_size": args.batch_size, "learning_rate": args.lr, "epochs": args.epochs, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}, indent=2), encoding="utf-8")
    clean_iter = iter(clean_loader); patched_iter = iter(patched_loader)
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(args.epochs):
        losses=[]
        for _ in range(max(len(clean_loader), len(patched_loader))):
            try: clean_batch=next(clean_iter)
            except StopIteration: clean_iter=iter(clean_loader); clean_batch=next(clean_iter)
            try: patched_batch=next(patched_iter)
            except StopIteration: patched_iter=iter(patched_loader); patched_batch=next(patched_iter)
            clean_images=clean_batch["img"].to(device); patched_images=patched_batch["img"].to(device)
            clean_batch={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in clean_batch.items()}
            patched_batch={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in patched_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if criterion is None:
                clean_loss,_=model.loss(clean_batch,model(clean_images)); patched_loss,_=model.loss(patched_batch,model(patched_images)); loss=0.5*(clean_loss.sum()+patched_loss.sum())
            else:
                clean_outputs=model(clean_images); patched_outputs=model(patched_images); loss=criterion(clean_outputs,clean_batch,patched_outputs,patched_batch["patch_mask"])["total"]
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at epoch={epoch+1}")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        mean_loss=sum(losses)/len(losses); record={"epoch":epoch+1,"train_loss":mean_loss,"train_batches":len(losses),"device":str(device)}
        with metrics_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
        torch.save({"model":model.state_dict(),"epoch":epoch+1,"train_loss":mean_loss},args.output/"last.pt")
        print(json.dumps(record))


if __name__ == "__main__":
    main()
