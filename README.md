# Adversarial Robust Person Detector

Research code for evaluating general COCO object-detector robustness to unseen
adversarial patches, while retaining the original person-focused experiment as
an auxiliary result.

## Scope

- Main Baseline: official COCO-pretrained YOLO11n (`yolo11n.pt`), no clean
  re-training, evaluated on all 80 COCO classes.
- Baseline + Adversarial Training: 80-class fine-tuning from the official
  pretrained weights with clean/synthetic-patch mixtures.
- Proposed RGB/Texture + Shape/Edge Model: 80-class shape-aware fusion.
- Auxiliary experiment: the completed person-only runs and checkpoints remain
  preserved under their existing `runs/` paths.
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

The main 80-class dataset configuration is `dataset_config.main80.yaml`.
Evaluate the official pretrained clean Baseline with
`python -m adversarial_robust_detector.evaluate_coco`; install
`pycocotools` first for official COCO AP/AP50/AP75 and per-class AP.

## Outputs

Each run may contain `last.pt`, `best_map.pt`, `best_loss.pt`, and
`metrics.jsonl`. Checkpoints and experiment outputs are ignored by default to
avoid accidentally committing large or private artifacts. A selected baseline
checkpoint may be copied to `artifacts/checkpoints/` and tracked with Git LFS.
