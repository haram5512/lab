"""Architecture and gradient diagnostics for Proposed RGB/Shape v3."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from .models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root.parent / "runs" / "main_80class" / "proposed_rgb_shape_v3_yolo_backbone"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProposedRGBShapeV3(ProposedV3Config(weights=str(root / "yolo11n.pt"))).to(device).train()
    images = torch.rand(2, 3, 640, 640, device=device)
    output = model(images)
    raw = output[0] if isinstance(output, tuple) else output
    tensors = []
    def collect(value):
        if isinstance(value, torch.Tensor): tensors.append(value)
        elif isinstance(value, dict):
            for child in value.values(): collect(child)
        elif isinstance(value, (tuple, list)):
            for child in value: collect(child)
    collect(raw)
    if not tensors: raise RuntimeError("diagnostic output contains no tensors")
    loss = sum(tensor.float().mean() for tensor in tensors)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite diagnostic output")
    loss.backward()
    shape_grad = sum(float(parameter.grad.detach().norm()) for parameter in model.shape_backbone.parameters() if parameter.grad is not None)
    alpha_grad = {f"grad_{name}": float(model.fusions[index].alpha.grad.detach().abs()) for index, name in enumerate(("alpha3", "alpha4", "alpha5"))}
    architecture = dict(model.initialization_report)
    architecture.update({"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "parameters_total": sum(parameter.numel() for parameter in model.parameters()), "shape_parameters": sum(parameter.numel() for parameter in model.shape_backbone.parameters()), "forward_diagnostics": model.last_diagnostics})
    diagnostics = {"forward_finite": all(bool(torch.isfinite(tensor).all()) for tensor in tensors), "loss_finite": bool(torch.isfinite(loss)), "shape_gradient_norm": shape_grad, "alpha_gradients": alpha_grad, "alpha_values": model.alpha_values, "output_shapes": [tuple(tensor.shape) for tensor in tensors], "checkpoint_round_trip": False}
    checkpoint = out / "v3_architecture_smoke.pt"
    torch.save({"model": model.state_dict(), "architecture": architecture}, checkpoint)
    restored = ProposedRGBShapeV3(ProposedV3Config(weights=str(root / "yolo11n.pt")))
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    diagnostics["checkpoint_round_trip"] = True
    (out / "architecture.json").write_text(json.dumps(architecture, indent=2), encoding="utf-8")
    (out / "diagnostic_results.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({"architecture": architecture, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
