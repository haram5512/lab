# Adversarial Robust Person Detector

Research code for evaluating whether person detection remains robust to unseen
adversarial patches and physical-world attacks by using RGB/texture and
shape/edge information.

## Scope

- Baseline Clean: COCO 2017 person detection with letterbox `640x640` input.
- Baseline + Adversarial Training: clean images mixed with synthetic patches.
- Proposed RGB/Texture + Shape/Edge Model: shape-aware feature fusion.
- Seen Patch and Unseen Patch evaluation: train patches and held-out patches are separated.
- Physical-world evaluation: AdvT-shirt-1K is not used for training or model selection.
- CrowdHuman is optional for occlusion and crowded-scene shape-branch experiments.

## Environment and data

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

COCO is intentionally not stored in GitHub. The portable default expects it at
`../datasets/coco`. For a different PC-specific path, copy
`dataset_config.local.example.yaml` to the ignored `dataset_config.local.yaml`.

## Smoke test and Baseline

```powershell
python -m adversarial_robust_detector.train_baseline --config dataset_config.local.yaml --max-images 100 --val-max-images 100 --epochs 1 --image-size 640 --batch-size 4 --lr 0.0001 --val-every 1 --num-workers 2 --pin-memory --persistent-workers --output runs\baseline_clean_smoke_100
```

The two-PC development/full-training workflow and current commands are in
`TWO_PC_WORKFLOW.md`. Each run records its source commit, GPU, dependencies,
data configuration, and training arguments in `run_metadata.json`.

## Outputs

Each run may contain `last.pt`, `best_map.pt`, `best_loss.pt`, and
`metrics.jsonl`. Checkpoints and experiment outputs are ignored by default to
avoid accidentally committing large or private artifacts. A selected baseline
checkpoint may be copied to `artifacts/checkpoints/` and tracked with Git LFS.
