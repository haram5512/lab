"""Generate reproducible gradient-optimized COCO adversarial patches."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
import yaml

from .attacks import EOTConfig, PatchConfig, PatchGenerator
from .baseline import BaselineDetector
from .dataset import Coco80DetectionDataset, CocoPersonConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("dataset_config.main80.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--strength", choices=("weak", "medium", "strong"), default="strong")
    parser.add_argument("--pool", choices=("train_seen", "unseen"), default="train_seen")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--source-images", type=int, default=50)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("artifacts/adversarial_patches"))
    parser.add_argument("--variant", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = config["train"]
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = root / "artifacts" / "checkpoints" / "best_map.pt"
        if not checkpoint.exists():
            candidates = sorted((root / "artifacts" / "checkpoints").glob("*.pt"))
            checkpoint = candidates[0] if candidates else None
    if checkpoint is None or not checkpoint.exists():
        raise SystemExit("No baseline checkpoint found; pass --checkpoint explicitly.")
    images = Path(split["images"])
    annotations = Path(split["annotations"])
    if not images.is_absolute():
        images = root / images
    if not annotations.is_absolute():
        annotations = root / annotations
    dataset = Coco80DetectionDataset(CocoPersonConfig(images, annotations, image_size=640, max_images=args.source_images))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = BaselineDetector("yolo11n.pt").to(device).eval()
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            detector.load_state_dict(state["model"], strict=False)
    except Exception as exc:
        raise SystemExit(f"Could not load baseline checkpoint {checkpoint}: {exc}") from exc
    defaults = {"weak": (100, 0.8, 1.2), "medium": (300, 0.7, 1.3), "strong": (500, 0.65, 1.35)}
    default_steps, scale_min, scale_max = defaults[args.strength]
    steps = args.steps or default_steps
    if args.variant == "v2":
        eot = EOTConfig(scale_min=scale_min, scale_max=scale_max, translate_fraction=0.14, rotation_degrees=25.0, perspective_jitter=0.12, brightness=0.30, contrast=0.30)
        seed = 7001 if args.pool == "train_seen" else 9101
    else:
        eot = EOTConfig(scale_min=scale_min, scale_max=scale_max)
        seed = 7 if args.pool == "train_seen" else 101
    generator = PatchGenerator(detector, PatchConfig(patch_size=args.patch_size, steps=steps, seed=seed, eot=eot), device=device)
    output_dir = args.output / args.variant / args.pool if args.variant == "v2" else args.output / args.pool
    start_index = 0
    source_samples = [dataset[index] for index in range(len(dataset))]
    images = [sample["img"].unsqueeze(0) for sample in source_samples]
    boxes = [sample["bboxes"][:1].unsqueeze(0) for sample in source_samples]
    classes = [sample["cls"][:1].squeeze(1).unsqueeze(0) for sample in source_samples]
    for index in range(args.count):
        # One reproducible target per image; the generator API remains open to
        # multi-object attacks later.
        result = generator.generate_many(images, boxes, classes, batch_size=8)
        patch_id = (chr(ord("A") + index) if args.pool == "train_seen" else chr(ord("F") + index)) + ("_v2" if args.variant == "v2" else "")
        path = output_dir / f"patch_{patch_id}.png"
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        generator.save(result, path, {"patch_id": patch_id, "pool": args.pool, "variant": args.variant, "git_commit": commit, "checkpoint": str(checkpoint), "source_detector": "YOLO11n", "dataset": "COCO train2017", "source_image_count": args.source_images, "heldout_split": "train2017_disjoint_subset", "target_policy": "all", "strength": args.strength, "steps": steps, "learning_rate": generator.config.learning_rate, "patch_size": args.patch_size, "seed": generator.config.seed, "attack_objective": "target_pre_nms_class_suppression", "initialization": "uniform_random_[0,1]", "eot": generator.config.eot.__dict__})
        print(f"saved={path} before={result.before_score:.8f} after={result.after_score:.8f}")


if __name__ == "__main__":
    main()
