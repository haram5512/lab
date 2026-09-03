# Shared research artifacts

Only reviewed, intentionally selected results belong here.

- `checkpoints/`: selected `best_map.pt`, optionally `last.pt` or `best_loss.pt`; `.pt` files use Git LFS.
- `results/<run-name>/`: `metrics.jsonl`, run config, `run_metadata.json`, and a short result summary.

Do not copy datasets, per-batch logs, caches, or every epoch checkpoint here.
