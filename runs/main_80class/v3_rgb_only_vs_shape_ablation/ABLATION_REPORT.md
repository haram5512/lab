# V3 RGB-only vs Shape Fusion Ablation

- Checkpoint: `C:\논문\runs\main_80class\proposed_rgb_shape_v3_full\best_seen_map.pt`
- Git commit: `a408805f54729e3cfbf729c74ce14ffa0e47fb74`
- Stored alpha: alpha3=0.137669, alpha4=0.236824, alpha5=0.243273
- RGB-only changes only the in-memory inference contribution to zero; the checkpoint was not modified.

## Overall metrics

| Mode | Condition | AP | AP50 | AP75 | AR100 | R50 | R75 | Precision | F1 | mIoU | GT-IoU | FP/img | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_rgb_only | clean | 0.5219 | 0.7630 | 0.5550 | 0.6372 | 0.6660 | 0.5372 | 0.7363 | 0.6994 | 0.8434 | 0.5617 | 0.5301 |  |
| v3_rgb_only | seen | 0.5076 | 0.7480 | 0.5359 | 0.6282 | 0.6524 | 0.5224 | 0.7265 | 0.6874 | 0.8416 | 0.5491 | 0.5458 | 0.0341 |
| v3_rgb_only | unseen | 0.5049 | 0.7456 | 0.5323 | 0.6270 | 0.6507 | 0.5193 | 0.7240 | 0.6854 | 0.8409 | 0.5472 | 0.5513 | 0.0347 |
| v3_normal | clean | 0.5225 | 0.7634 | 0.5569 | 0.6396 | 0.6692 | 0.5406 | 0.7360 | 0.7010 | 0.8430 | 0.5642 | 0.5333 |  |
| v3_normal | seen | 0.5270 | 0.7588 | 0.5634 | 0.6428 | 0.6747 | 0.5489 | 0.7137 | 0.6936 | 0.8465 | 0.5711 | 0.6014 | 0.0194 |
| v3_normal | unseen | 0.5327 | 0.7617 | 0.5699 | 0.6456 | 0.6803 | 0.5576 | 0.7136 | 0.6966 | 0.8483 | 0.5771 | 0.6066 | 0.0196 |

## Per-GT transitions

| Condition | Both Success | Shape Helps | Shape Hurts | Both Fail | Net Benefit | Net Rate |
|---|---:|---:|---:|---:|---:|---:|
| clean | 7182 | 182 | 147 | 3493 | 35 | 0.3181% |
| seen | 7021 | 403 | 158 | 3422 | 245 | 2.2265% |
| unseen | 7006 | 480 | 154 | 3364 | 326 | 2.9626% |

## Verdict

ADAPTIVE ALPHA POTENTIALLY USEFUL

See the CSV/JSON files and `shape_ablation_visuals` for complete quantitative and qualitative evidence.
