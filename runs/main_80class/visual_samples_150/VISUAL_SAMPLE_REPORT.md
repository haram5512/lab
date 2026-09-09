# Visual sample report

Generated exactly 150 images from existing COCO data, patch pools, and checkpoints.

- Git commit: `a408805f54729e3cfbf729c74ce14ffa0e47fb74`
- Device: `cuda`
- Training pairs: 50 clean + 50 perturbed (same COCO IDs)
- Test comparisons: 50 (clean=10, seen=15, unseen=25)
- Patch transformations: existing torso placement and configured EOT/randomization; no patch optimization performed.

## Recommended first five test files

- `test_001.jpg` — CLEAN: v3 success / adv failure (COCO image 17207)
- `test_002.jpg` — CLEAN: v3 success / adv failure (COCO image 19221)
- `test_003.jpg` — CLEAN: adv success / v3 failure (COCO image 17379)
- `test_004.jpg` — CLEAN: v3 higher matched IoU (COCO image 36494)
- `test_005.jpg` — CLEAN: v3 higher matched IoU (COCO image 872)

Sanity checks are in `train_sanity_checks.json` and `test_sanity_checks.json`. Test inference used a deterministic 360-image candidate pool per condition; the rendered set is the requested 50 representative cases.
