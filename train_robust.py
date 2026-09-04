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
from .coco_eval_core import evaluate_model


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
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-max-images", type=int, default=0)
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
    val_split = config["validation"]["clean"]
    val_images, val_annotations = Path(val_split["images"]), Path(val_split["annotations"])
    if not val_images.is_absolute(): val_images = root / val_images
    if not val_annotations.is_absolute(): val_annotations = root / val_annotations
    if not 0.0 < args.patch_probability < 1.0: raise ValueError("--patch-probability must be between 0 and 1")
    common = {"image_size": args.image_size, "max_images": None if args.max_images <= 0 else args.max_images, "target_policy": "person", "required_category_ids": (1,)}
    ds_config = CocoPersonConfig(images, annotations, patch_mode="clean", **common)
    clean_ds = Coco80DetectionDataset(ds_config)
    patched_ds = Coco80DetectionDataset(CocoPersonConfig(images, annotations, patch_mode="seen", patch_dirs=(patch_dir,), patch_probability=1.0, placement_mode="torso", torso_relative_y=.38, position_jitter=.015, patch_scale_min=.95, patch_scale_max=1.05, patch_rotation_degrees=3, patch_brightness_jitter=.03, patch_perspective_jitter=.005, **common))
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coco_person_collate, num_workers=args.num_workers)
    patched_loader = DataLoader(patched_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coco_person_collate, num_workers=args.num_workers)
    val_limit = None if args.val_max_images <= 0 else args.val_max_images
    val_common = {"image_size": args.image_size, "max_images": val_limit, "target_policy": "person", "required_category_ids": (1,)}
    val_clean_ds = Coco80DetectionDataset(CocoPersonConfig(val_images, val_annotations, patch_mode="clean", **val_common))
    val_seen_ds = Coco80DetectionDataset(CocoPersonConfig(val_images, val_annotations, patch_mode="seen", patch_dirs=(patch_dir,), patch_probability=1.0, placement_mode="torso", torso_relative_y=.38, position_jitter=.015, patch_scale_min=.95, patch_scale_max=1.05, patch_rotation_degrees=3, patch_brightness_jitter=.03, patch_perspective_jitter=.005, **val_common))
    val_clean_loader = DataLoader(val_clean_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coco_person_collate, num_workers=args.num_workers)
    val_seen_loader = DataLoader(val_seen_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coco_person_collate, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineDetector(args.weights) if args.model == "baseline" else ShapeAwareYolo(ModelConfig(weights=args.weights, num_classes=None))
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = RobustTrainingLoss(model) if args.model == "proposed" else None
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (args.output / "run_metadata.json").write_text(json.dumps({"git_commit": commit, "git_dirty": bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()), "model": args.model, "source_checkpoint": args.weights, "dataset": "COCO 2017 person-containing images, 80-class targets", "patch_pool": str(patch_dir.resolve()), "patch_ids": [p.name for p in patched_ds.patch_files], "patch_profile": "v5-C torso/very-mild", "validation": "clean+train_seen only", "clean_patch_ratio": [1-args.patch_probability, args.patch_probability], "image_size": args.image_size, "batch_size": args.batch_size, "effective_images_per_step": args.batch_size*2, "learning_rate": args.lr, "optimizer":"AdamW", "epochs": args.epochs, "val_every": args.val_every, "val_max_images": args.val_max_images, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "torch":torch.__version__, "cuda":torch.version.cuda}, indent=2), encoding="utf-8")
    clean_iter = iter(clean_loader); patched_iter = iter(patched_loader)
    metrics_path = args.output / "metrics.jsonl"
    best_clean = float("-inf")
    best_seen = float("-inf")
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
                clean_loss,_=model.loss(clean_batch,model(clean_images)); patched_loss,_=model.loss(patched_batch,model(patched_images)); loss=(1-args.patch_probability)*clean_loss.sum()+args.patch_probability*patched_loss.sum()
            else:
                clean_outputs=model(clean_images); patched_outputs=model(patched_images); loss=criterion(clean_outputs,clean_batch,patched_outputs,patched_batch["patch_mask"])["total"]
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at epoch={epoch+1}")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        mean_loss=sum(losses)/len(losses); record={"epoch":epoch+1,"train_loss":mean_loss,"train_batches":len(losses),"clean_samples":len(clean_ds),"perturbed_samples":len(patched_ds),"patch_ratio":args.patch_probability,"device":str(device),"max_vram_gib":torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0}
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            clean_metrics = evaluate_model(model, val_clean_loader, val_annotations, val_clean_ds, device, args.image_size, "clean")
            seen_metrics = evaluate_model(model, val_seen_loader, val_annotations, val_seen_ds, device, args.image_size, "seen")
            record.update({"clean_ap": clean_metrics["ap"], "clean_ap50": clean_metrics["ap50"], "clean_ap75": clean_metrics["ap75"], "clean_person_ap": clean_metrics["person_ap"], "seen_ap": seen_metrics["ap"], "seen_ap50": seen_metrics["ap50"], "seen_ap75": seen_metrics["ap75"], "seen_person_ap": seen_metrics["person_ap"], "seen_attacked_object_detection_rate": seen_metrics["attacked_object_detection_rate"], "val_seconds": clean_metrics["seconds"] + seen_metrics["seconds"]})
            state = {"model": model.state_dict(), "epoch": epoch + 1, "train_loss": mean_loss, "clean": clean_metrics, "seen": seen_metrics, "git_commit": commit}
            if clean_metrics["ap"] > best_clean:
                best_clean = clean_metrics["ap"]; torch.save(state, args.output / "best_clean_map.pt")
            if seen_metrics["ap"] > best_seen:
                best_seen = seen_metrics["ap"]; torch.save(state, args.output / "best_seen_map.pt")
            model.train()
        with metrics_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
        torch.save({"model":model.state_dict(),"epoch":epoch+1,"train_loss":mean_loss,"git_commit":commit},args.output/"last.pt")
        print(json.dumps(record))


if __name__ == "__main__":
    main()
