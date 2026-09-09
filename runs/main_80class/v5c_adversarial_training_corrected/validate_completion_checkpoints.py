import json
import sys

import torch


epochs = {}
for checkpoint_path in sys.argv[1:]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid checkpoint structure: {checkpoint_path}")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch <= 0:
        raise ValueError(f"Invalid checkpoint epoch: {checkpoint_path}: {epoch}")
    epochs[checkpoint_path] = epoch

print(json.dumps(epochs))
