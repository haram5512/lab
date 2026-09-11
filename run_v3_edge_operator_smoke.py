"""Pre-training visual, initialization, forward, and backward checks for v3 edge ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config, build_edge_extractor


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "main_80class" / "v3_edge_operator_ablation" / "smoke"
OPERATORS = ("sobel", "canny", "laplacian")


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_edge(edge: torch.Tensor, path: Path) -> None:
    array = edge.detach().squeeze().clamp(0, 1).mul(255).byte().cpu().numpy()
    Image.fromarray(array, mode="L").save(path)


def tensors_in(value: object) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        result.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            result.extend(tensors_in(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            result.extend(tensors_in(child))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val = config["validation"]["clean"]
    images = (ROOT / val["images"]).resolve()
    annotations = (ROOT / val["annotations"]).resolve()
    dataset = Coco80DetectionDataset(CocoPersonConfig(
        images_dir=images, annotations_file=annotations, image_size=640,
        max_images=5, patch_mode="clean", patch_probability=0.0, patch_dirs=(),
        target_policy="person", required_category_ids=(1,), seed=42,
    ))
    samples = [dataset[index] for index in range(5)]
    edge_metadata: dict[str, object] = {}
    for operator in OPERATORS:
        operator_dir = OUT / "edge_maps" / operator
        operator_dir.mkdir(parents=True)
        extractor, metadata = build_edge_extractor(ProposedV3Config(edge_operator=operator))
        edge_metadata[operator] = metadata
        with torch.inference_mode():
            for index, sample in enumerate(samples, 1):
                edge = extractor(sample["img"].unsqueeze(0))
                if edge.shape != (1, 1, 640, 640) or not torch.isfinite(edge).all():
                    raise RuntimeError(f"invalid {operator} edge tensor: {tuple(edge.shape)}")
                if float(edge.min()) < 0.0 or float(edge.max()) > 1.0:
                    raise RuntimeError(f"{operator} edge tensor is outside [0,1]")
                save_edge(edge, operator_dir / f"edge_{index:02d}.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnostics: dict[str, object] = {}
    digests: dict[str, str] = {}
    for operator in OPERATORS:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        model = ProposedRGBShapeV3(ProposedV3Config(
            weights=str(ROOT / "yolo11n.pt"), edge_operator=operator
        )).to(device).train()
        digests[operator] = parameter_digest(model)
        batch_results = []
        for offset in (0, 2):
            batch = coco_person_collate(samples[offset:offset + 2])
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            model.zero_grad(set_to_none=True)
            predictions = model(batch["img"])
            loss, _ = model.loss(batch, predictions)
            scalar_loss = loss.sum()
            if not torch.isfinite(scalar_loss):
                raise FloatingPointError(f"non-finite {operator} smoke loss")
            scalar_loss.backward()
            shape_gradient = sum(
                float(parameter.grad.detach().norm())
                for parameter in model.shape_backbone.parameters() if parameter.grad is not None
            )
            outputs = tensors_in(predictions)
            batch_results.append({
                "loss": float(scalar_loss.detach()),
                "output_finite": all(bool(torch.isfinite(value).all()) for value in outputs),
                "shape_gradient_norm": shape_gradient,
                "alpha_gradients": {
                    f"alpha{index}": float(fusion.alpha.grad.detach().abs())
                    for index, fusion in enumerate(model.fusions, 3)
                },
                "forward_diagnostics": model.last_diagnostics,
            })
        diagnostics[operator] = {
            "edge_preprocessing": model.edge_preprocessing,
            "initial_parameter_digest": digests[operator],
            "batches": batch_results,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(set(digests.values())) != 1:
        raise RuntimeError(f"trainable initialization differs across edge operators: {digests}")
    report = {
        "status": "passed",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "edge_samples_per_operator": 5,
        "initial_trainable_parameters_identical": True,
        "edge_metadata": edge_metadata,
        "diagnostics": diagnostics,
    }
    (OUT / "smoke_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
