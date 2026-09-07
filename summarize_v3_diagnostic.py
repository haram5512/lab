from __future__ import annotations
import json
from pathlib import Path

def main() -> None:
    root=Path(__file__).resolve().parent; out=root.parent/"runs"/"main_80class"/"proposed_rgb_shape_v3_yolo_backbone"; metrics=[json.loads(line) for line in (out/"metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()];
    alpha=[{"epoch":m["epoch"],**m["fusion_alpha"]} for m in metrics if isinstance(m.get("fusion_alpha"),dict)]
    grad=[{"epoch":m["epoch"],"shape_gradient_norm":m.get("shape_gradient_norm")} for m in metrics]
    speed=json.loads((out/"optimization_benchmark.json").read_text(encoding="utf-8"))
    (out/"alpha_history.json").write_text(json.dumps(alpha,indent=2),encoding="utf-8")
    (out/"gradient_history.json").write_text(json.dumps(grad,indent=2),encoding="utf-8")
    (out/"speed_profile.json").write_text(json.dumps({"diagnostic_epoch_seconds": [m.get("epoch_seconds") for m in metrics],"diagnostic_images_per_second": [m.get("images_per_second") for m in metrics],"benchmark":speed},indent=2),encoding="utf-8")
    print(json.dumps({"epochs":len(metrics),"alpha_last":alpha[-1],"gradient_last":grad[-1],"speed":speed},indent=2))
if __name__=="__main__": main()
