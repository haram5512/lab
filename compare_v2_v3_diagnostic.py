"""Evaluate v2 and a v3 diagnostic checkpoint on the same small subset."""
from __future__ import annotations

import json
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

from .coco_eval_core import evaluate_model
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape_v2 import ProposedRGBShapeV2, ProposedV2Config
from .models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config


def load(model, checkpoint: Path, device):
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    return model.to(device).eval()


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root.parent / "runs" / "main_80class" / "proposed_rgb_shape_v3_yolo_backbone"
    config = yaml.safe_load((root / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = config["validation"]["clean"]
    images = (root / Path(val["images"])).resolve(); annotations = (root / Path(val["annotations"])).resolve()
    payload = json.loads(annotations.read_text(encoding="utf-8")); categories = tuple(sorted(int(item["id"]) for item in payload["categories"]))
    common = dict(images_dir=images, annotations_file=annotations, image_size=640, max_images=500, target_policy="person", required_category_ids=(), category_ids=categories, seed=7, position_jitter=.015, placement_mode="torso", torso_relative_y=.38, patch_scale_min=.95, patch_scale_max=1.05, patch_rotation_degrees=3., patch_brightness_jitter=.03, patch_perspective_jitter=.005)
    seen_patch = (root / "runs" / "v5_diagnostic" / "patch_V5-C.png",)
    datasets = {"clean": Coco80DetectionDataset(CocoPersonConfig(**common, patch_mode="clean", patch_probability=0., patch_dirs=())), "seen": Coco80DetectionDataset(CocoPersonConfig(**common, patch_mode="seen", patch_probability=1., patch_dirs=seen_patch))}
    loaders = {mode: DataLoader(ds, batch_size=16, shuffle=False, collate_fn=coco_person_collate, num_workers=0) for mode, ds in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v2_path = root.parent / "runs" / "main_80class" / "proposed_rgb_shape_v2_full" / "best_seen_map.pt"
    v3_path = out / "v3_diagnostic_last.pt"
    models = {"v2_best_seen": load(ProposedRGBShapeV2(ProposedV2Config(weights=str(root / "yolo11n.pt"))), v2_path, device), "v3_diagnostic": load(ProposedRGBShapeV3(ProposedV3Config(weights=str(root / "yolo11n.pt"))), v3_path, device)}
    results = {}
    for name, model in models.items():
        results[name] = {mode: evaluate_model(model, loaders[mode], annotations, datasets[mode], device, 640, mode, confidence=.001, iou_threshold=.7) for mode in ("clean", "seen")}
    for values in results.values():
        for metric in values.values(): values[metric].pop("person_match_status", None)
    (out / "diagnostic_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
