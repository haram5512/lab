# Two-PC research workflow

GitHub Private Repository is the source of truth for code, shared configuration,
selected checkpoints, metrics, and concise result summaries. COCO and ordinary
`runs/` contents remain local to each PC.

## Roles

### RTX 5070 Ti: full experiments

Use this PC for full COCO training/evaluation, adversarial and proposed-model
training, Seen/Unseen evaluation, AdvT-shirt-1K evaluation, and large benchmarks.
Record `git rev-parse HEAD` before starting. Do not pull or change the checked-out
code while a run is active; develop on the RTX 3050 meanwhile and pull only after
the run ends.

The verified Baseline Clean runtime is batch 16, workers 0, CUDA, learning rate
`1e-4`, 100 epochs, and validation every 5 epochs. Common research settings live
in `baseline_train_config.yaml`; runtime choices stay in the command line.

The new main experiment uses `dataset_config.main80.yaml` and the official
pretrained `yolo11n.pt` without clean re-training. The old person-only config
and runs remain auxiliary and are not overwritten.

## Gradient patch generation and defense training

Patch generation is a separate phase. The detector is frozen and only the
patch tensor is optimized against the differentiable pre-NMS YOLO output. The
generator applies EOT scale, translation, rotation, appearance, and bounded
affine-perspective jitter, then writes a PNG plus JSON metadata.

Generate Train/Seen patches from `train2017` only:

```powershell
python -m adversarial_robust_detector.generate_patches --pool train_seen --count 5 --source-images 50 --steps 200 --output artifacts\adversarial_patches
```

Generate a separate Unseen pool with a different seed:

```powershell
python -m adversarial_robust_detector.generate_patches --pool unseen --count 3 --source-images 50 --steps 250 --output artifacts\adversarial_patches
```

Never pass the unseen pool to defense training. The 80-class fine-tuning entry
point uses clean and Train/Seen samples in a 50/50 pair:

```powershell
python -m adversarial_robust_detector.train_robust --model baseline --patch-dir artifacts\adversarial_patches\train_seen --config dataset_config.main80.yaml --max-images 0 --epochs 100 --image-size 640 --batch-size 16 --lr 0.0001 --num-workers 0 --output runs\main_80class\adversarial_training
python -m adversarial_robust_detector.train_robust --model proposed --patch-dir artifacts\adversarial_patches\train_seen --config dataset_config.main80.yaml --max-images 0 --epochs 100 --image-size 640 --batch-size 8 --lr 0.0001 --num-workers 0 --output runs\main_80class\proposed_rgb_shape
```

Run the 1-2 epoch smoke versions first with `--max-images 50 --epochs 1`.

### RTX 3050: development

Use this PC for code/model/DataLoader/evaluator changes, tests, documentation,
dummy forward/backward, inference, and smoke runs of at most 50-100 images. Do
not start full 100-epoch COCO training by default. Start with batch 2 or 4 and
adjust only runtime arguments, never model structure.

## Per-PC data configuration

The tracked `dataset_config.yaml` assumes `../datasets/coco`. For another local
path, copy `dataset_config.local.example.yaml` to `dataset_config.local.yaml`,
edit only that ignored file, and pass `--config dataset_config.local.yaml`.

## Required Git sequence

Before work:

```powershell
git status
git pull --ff-only
git lfs pull
```

After code/tests:

```powershell
git status
git add <reviewed-files>
git commit -m "Describe the research change"
git push
```

On the other PC, run `git pull --ff-only` before starting. Use `main` while the
workflow is simple; use a `dev` or feature branch when both PCs must change
overlapping files. Never commit directly from both PCs without pulling first.

## Development gate on RTX 3050

```powershell
python -m pytest -q
python -m adversarial_robust_detector.train_baseline --config dataset_config.local.yaml --max-images 100 --val-max-images 100 --epochs 1 --image-size 640 --batch-size 4 --lr 0.0001 --val-every 1 --num-workers 2 --pin-memory --persistent-workers --output runs\baseline_clean_smoke_100
```

Push only after this gate passes. Then pull the exact commit on the RTX 5070 Ti.

## Full Baseline on RTX 5070 Ti

```powershell
git rev-parse HEAD
python -m adversarial_robust_detector.train_baseline --config dataset_config.yaml --weights yolo11n.pt --max-images 0 --val-max-images 0 --epochs 100 --image-size 640 --batch-size 16 --lr 0.0001 --val-every 5 --num-workers 0 --output runs\baseline_clean_full_letterbox
```

For the new no-retraining 80-class clean Baseline evaluation, use
`evaluate_coco.py` with `dataset_config.main80.yaml`; it requires
`pycocotools` and writes official COCO metrics, including per-class and person
AP. Do not use the old person-only `train_baseline` command as the main result.

Every new run automatically writes `run_metadata.json` with the commit, model,
dataset, preprocessing, optimizer settings, GPU, PyTorch/CUDA versions, and
runtime parameters. `metrics.jsonl` contains the per-epoch metrics.

## Sharing selected results

After a run, review its files and copy only selected artifacts:

```powershell
Copy-Item runs\<run>\best_map.pt artifacts\checkpoints\<descriptive-name>-best_map.pt
Copy-Item runs\<run>\last.pt artifacts\checkpoints\<descriptive-name>-last.pt  # only if needed
Copy-Item runs\<run>\metrics.jsonl artifacts\results\<run>\metrics.jsonl
Copy-Item runs\<run>\run_metadata.json artifacts\results\<run>\run_metadata.json
git add artifacts
git commit -m "Add selected <run> results"
git push
```

The RTX 3050 retrieves checkpoint contents with `git pull --ff-only` followed by
`git lfs pull`, then uses `map_location=device`, `model.eval()`, and a small
inference/evaluator smoke test. Keep every-epoch checkpoints and large logs local.
