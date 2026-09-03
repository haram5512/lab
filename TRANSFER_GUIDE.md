# Transfer guide

This repository contains code and configuration, not COCO images or other large datasets.

## Windows setup

```powershell
git clone <PRIVATE_REPOSITORY_URL>
cd adversarial_robust_detector
py -3.10 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a different GPU, install the PyTorch build appropriate for that machine before
installing the remaining requirements. Verify with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Download COCO 2017 separately. The tracked configuration expects
`../datasets/coco`; use the ignored `dataset_config.local.yaml` for any other
machine-specific absolute path.

## Smoke test

```powershell
python -m adversarial_robust_detector.train_baseline --config dataset_config.local.yaml --max-images 100 --val-max-images 100 --epochs 1 --image-size 640 --batch-size 4 --lr 0.0001 --val-every 1 --num-workers 2 --pin-memory --persistent-workers --output runs\\baseline_clean_smoke_100
```

## Continue the clean Baseline

The RTX 5070 Ti verified configuration is batch 16, workers 0, learning rate
`1e-4`, letterbox `640`, and validation every 5 epochs:

```powershell
python -m adversarial_robust_detector.train_baseline --config dataset_config.yaml --max-images 0 --val-max-images 0 --epochs 100 --image-size 640 --batch-size 16 --lr 0.0001 --val-every 5 --num-workers 0 --output runs\\baseline_clean_full_letterbox
```

Metrics are written to `metrics.jsonl`; `last.pt`, `best_map.pt`, and
`best_loss.pt` are the available checkpoint roles. Copy only reviewed artifacts
to `artifacts/`; other experiment outputs remain local. See
`TWO_PC_WORKFLOW.md` for Git and Git LFS synchronization.

## Research stages

- Baseline Clean: COCO person detection with letterbox preprocessing.
- Baseline + Adversarial Training: clean and synthetic patch mixing.
- Proposed RGB/Texture + Shape/Edge branches: shape-aware fusion model.
- Seen and Unseen Patch evaluation: separate train and evaluation patch sets.
- Physical-world evaluation: AdvT-shirt-1K, held out from training and model selection.
