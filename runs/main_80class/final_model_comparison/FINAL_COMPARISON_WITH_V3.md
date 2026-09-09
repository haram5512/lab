# Final Comparison Including Proposed RGB/Shape v3

Evaluation: COCO val2017 person-containing set, 11,004 person GT, 640 letterbox, same corrected evaluator and patch pools.

## Representative best-seen comparison

| Model | Clean AP | Seen AP | Unseen AP | Seen R50 | Unseen R50 | Seen GT-IoU | Unseen GT-IoU | Seen Failure | Unseen Failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adv baseline | 0.4904 | 0.4964 | 0.4948 | 0.6891 | 0.6821 | 0.5819 | 0.5723 | 0.0169 | 0.0178 |
| Proposed v2 | 0.5253 | 0.5120 | 0.5146 | 0.6559 | 0.6590 | 0.5572 | 0.5600 | 0.0296 | 0.0287 |
| Proposed v3 | 0.5231 | 0.5285 | 0.5343 | 0.6776 | 0.6834 | 0.5766 | 0.5829 | 0.0176 | 0.0176 |

## Interpretation

Final verdict: **GO**.

Proposed v3 improves unseen AP, Recall@0.50/0.75, Precision, F1, GT-IoU, and Failure Rate over the conventional adversarial baseline. Seen Recall and GT-IoU remain slightly below the Adv baseline, so this is GO for unseen robustness rather than STRONG GO across every metric.

Full rows: `all_models_metrics_with_v3.csv`; v3-only rows: `proposed_rgb_shape_v3_final_eval/v3_metrics.csv`.
