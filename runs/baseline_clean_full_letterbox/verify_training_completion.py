import json
import sys
from pathlib import Path

import torch


output = Path(__file__).resolve().parent
metrics_path = output / "metrics.jsonl"
error_path = output / "train.error.log"
early_stop_path = output / "early_stop.log"

if not metrics_path.is_file():
    raise SystemExit("metrics.jsonl missing")

records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not records:
    raise SystemExit("metrics.jsonl empty")

latest_epoch = int(records[-1]["epoch"])
early_stopped = early_stop_path.is_file() and "early stopping:" in early_stop_path.read_text(encoding="utf-8")
if latest_epoch != 100 and not early_stopped:
    raise SystemExit(f"training incomplete without valid early stop: epoch={latest_epoch}")

if error_path.is_file() and error_path.read_text(encoding="utf-8", errors="replace").strip():
    raise SystemExit("train.error.log is not empty")

loaded = {}
for name in ("last.pt", "best.pt", "best_map.pt"):
    path = output / name
    if not path.is_file() or path.stat().st_size < 1_000_000:
        raise SystemExit(f"{name} missing or incomplete")
    loaded[name] = torch.load(path, map_location="cpu", weights_only=False)

if int(loaded["last.pt"].get("epoch", -1)) != latest_epoch:
    raise SystemExit("last.pt epoch does not match final metrics record")

best_metric = max(float(record["map"]) for record in records if record.get("map") is not None)
for name in ("best.pt", "best_map.pt"):
    checkpoint_map = loaded[name].get("map")
    if checkpoint_map is None or abs(float(checkpoint_map) - best_metric) > 1e-9:
        raise SystemExit(f"{name} does not contain the best recorded AP")

print(f"VERIFIED epoch={latest_epoch} best_map={best_metric:.6f} early_stopped={early_stopped}")
sys.exit(0)
