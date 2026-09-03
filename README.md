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

COCO is intentionally not stored in GitHub. Download COCO 2017 separately and
place `train2017`, `val2017`, and the annotation JSON files under
`datasets\coco\`. Set local paths in `dataset_config.yaml`; paths are relative
to the project root where possible.

## Smoke test and Baseline

```powershell
python train_baseline.py --max-images 100 --val-max-images 20 --epochs 3 --image-size 640 --batch-size 8 --lr 0.0001 --val-every 5 --num-workers 2 --pin-memory --persistent-workers --output runs\baseline_clean_smoke
```

The current full clean Baseline command is documented in
`baseline_train_config.yaml` and `TRANSFER_GUIDE.md`. It uses COCO train/val
person annotations, batch size 8, two DataLoader workers, pinned/persistent
workers, learning rate `1e-4`, 100 epochs, and validation every 5 epochs.

## Outputs

Each run may contain `last.pt`, `best_map.pt`, `best_loss.pt`, and
`metrics.jsonl`. Checkpoints and experiment outputs are ignored by default to
avoid accidentally committing large or private artifacts. A selected baseline
checkpoint may be tracked with Git LFS when needed.
