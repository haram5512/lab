"""Build final tables, deltas, alpha comparison, and verdict for the v3 edge ablation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "main_80class"
OUT = RUNS / "v3_edge_operator_ablation"
RUN_NAMES = {
    "sobel": "proposed_rgb_shape_v3_full",
    "canny": "proposed_v3_canny_30ep",
    "laplacian": "proposed_v3_laplacian_30ep",
}
LABELS = {
    "person_ap": "AP", "person_ap50": "AP50", "person_ap75": "AP75",
    "person_ar100": "AR100", "person_recall_iou50_conf025": "R50",
    "person_recall_iou75_conf025": "R75", "precision_iou50_conf025": "Precision",
    "f1_iou50_conf025": "F1", "mean_matched_iou": "Matched-IoU",
    "mean_gt_iou_with_misses": "GT-IoU", "fp_per_image_conf025": "FP/img",
    "failure_rate": "Failure",
}


def read_rows() -> list[dict[str, Any]]:
    sobel_path = RUNS / "proposed_rgb_shape_v3_final_eval" / "v3_metrics.csv"
    sobel = [row for row in csv.DictReader(sobel_path.open(encoding="utf-8"))
             if row["checkpoint"] == "best_seen_map"]
    rows: list[dict[str, Any]] = []
    for row in sobel:
        rows.append({"edge_operator": "sobel", "checkpoint_epoch": 5, **row})
    edge_path = OUT / "canny_laplacian_final_metrics.csv"
    rows.extend(csv.DictReader(edge_path.open(encoding="utf-8")))
    return rows


def numeric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        item = dict(row)
        item["checkpoint_epoch"] = int(item["checkpoint_epoch"])
        for field in LABELS:
            item[field] = None if row.get(field, "") in ("", None) else float(row[field])
        converted.append(item)
    return converted


def alpha_values(operator: str) -> dict[str, float]:
    checkpoint = RUNS / RUN_NAMES[operator] / "best_seen_map.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    return {f"alpha{index + 3}": float(state[f"fusions.{index}.alpha"]) for index in range(3)}


def best(rows: list[dict[str, Any]], condition: str, metric: str, lower: bool = False) -> str:
    subset = [row for row in rows if row["condition"] == condition and row[metric] is not None]
    chosen = min(subset, key=lambda row: row[metric]) if lower else max(subset, key=lambda row: row[metric])
    return chosen["edge_operator"]


def md_table(headers: list[str], values: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in values:
        formatted = ["" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value)) for value in row]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = numeric_rows(read_rows())
    order = {name: index for index, name in enumerate(("sobel", "canny", "laplacian"))}
    rows.sort(key=lambda row: (order[row["edge_operator"]], ("clean", "seen", "unseen").index(row["condition"])))
    fields = ["edge_operator", "condition", "checkpoint_epoch", *LABELS]
    with (OUT / "edge_operator_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["edge_operator"], row["condition"]): row for row in rows}
    delta_metrics = (
        "person_ap", "person_recall_iou50_conf025", "person_recall_iou75_conf025",
        "precision_iou50_conf025", "mean_gt_iou_with_misses", "fp_per_image_conf025", "failure_rate",
    )
    deltas = []
    for operator in ("canny", "laplacian"):
        for condition in ("clean", "seen", "unseen"):
            current, sobel = by_key[(operator, condition)], by_key[("sobel", condition)]
            deltas.append({
                "edge_operator": operator, "condition": condition,
                **{f"delta_{metric}": None if current[metric] is None or sobel[metric] is None else current[metric] - sobel[metric]
                   for metric in delta_metrics},
            })
    with (OUT / "deltas_vs_sobel.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)

    alphas = [{"edge_operator": operator, **alpha_values(operator)} for operator in RUN_NAMES]
    with (OUT / "alpha_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alphas[0]))
        writer.writeheader()
        writer.writerows(alphas)

    latency = json.loads((OUT / "latency_benchmark.json").read_text(encoding="utf-8"))["results"]
    answers = {
        "best_clean_ap": best(rows, "clean", "person_ap"),
        "best_seen_ap": best(rows, "seen", "person_ap"),
        "best_unseen_ap": best(rows, "unseen", "person_ap"),
        "best_unseen_r50": best(rows, "unseen", "person_recall_iou50_conf025"),
        "best_unseen_r75": best(rows, "unseen", "person_recall_iou75_conf025"),
        "lowest_seen_failure": best(rows, "seen", "failure_rate", lower=True),
    }
    tradeoff_scores = {}
    for operator in RUN_NAMES:
        unseen = by_key[(operator, "unseen")]
        precision_rank = sorted(RUN_NAMES, key=lambda name: by_key[(name, "unseen")]["precision_iou50_conf025"], reverse=True).index(operator) + 1
        fp_rank = sorted(RUN_NAMES, key=lambda name: by_key[(name, "unseen")]["fp_per_image_conf025"]).index(operator) + 1
        tradeoff_scores[operator] = precision_rank + fp_rank
    answers["best_precision_fp_tradeoff"] = min(tradeoff_scores, key=tradeoff_scores.get)

    # Transparent equal-rank overall score: robustness first, then clean quality and speed.
    ranking_specs = [
        ("unseen", "person_ap", False), ("unseen", "person_recall_iou50_conf025", False),
        ("unseen", "person_recall_iou75_conf025", False), ("seen", "failure_rate", True),
        ("seen", "person_ap", False), ("clean", "person_ap", False),
    ]
    overall_scores = {operator: 0 for operator in RUN_NAMES}
    for condition, metric, lower in ranking_specs:
        ranked = sorted(RUN_NAMES, key=lambda name: by_key[(name, condition)][metric], reverse=not lower)
        for rank, operator in enumerate(ranked, 1):
            overall_scores[operator] += rank
    speed_ranked = sorted(RUN_NAMES, key=lambda name: latency[name]["model_including_edge_plus_nms"]["median_ms"])
    for rank, operator in enumerate(speed_ranked, 1):
        overall_scores[operator] += rank
    overall = min(overall_scores, key=overall_scores.get)
    answers["best_speed"] = speed_ranked[0]
    answers["best_overall"] = overall

    comparison_values = [[row["edge_operator"], row["condition"], *[row[field] for field in LABELS]] for row in rows]
    delta_values = [[row["edge_operator"], row["condition"], *[row[f"delta_{field}"] for field in delta_metrics]] for row in deltas]
    alpha_values_table = [[row["edge_operator"], row["alpha3"], row["alpha4"], row["alpha5"]] for row in alphas]
    speed_values = [[operator,
                     latency[operator]["edge_preprocessing_gpu"]["median_ms"],
                     latency[operator]["model_including_edge_plus_nms"]["median_ms"],
                     latency[operator]["model_including_edge_plus_nms"]["mean_ms"],
                     latency[operator]["model_including_edge_plus_nms"]["fps_from_median"]]
                    for operator in RUN_NAMES]
    report = f"""# v3 Edge Operator Controlled Ablation

Only the Shape-input edge operator differs. RGB/Shape backbones, P3/P4/P5 fusion, initialization, data, patches, EOT, optimizer, learning rate, batch size, image size, seed, evaluator, and checkpoint rule were held fixed. Canny and Laplacian each ran exactly 30 epochs; Unseen data was never used for checkpoint selection.

## Full best-seen comparison

{md_table(["Edge", "Condition", *LABELS.values()], comparison_values)}

## Delta versus Sobel (operator - Sobel)

{md_table(["Edge", "Condition", *[LABELS[field] for field in delta_metrics]], delta_values)}

## Fusion alpha at best-seen checkpoint

{md_table(["Edge", "alpha3", "alpha4", "alpha5"], alpha_values_table)}

## RTX 5070 Ti latency, batch 1, 640x640, FP32

{md_table(["Edge", "Edge median ms", "Model+NMS median ms", "Model+NMS mean ms", "FPS"], speed_values)}

## Decisions

- Highest Clean AP: {answers['best_clean_ap'].upper()}
- Highest Seen AP: {answers['best_seen_ap'].upper()}
- Highest Unseen AP: {answers['best_unseen_ap'].upper()}
- Highest Unseen R50: {answers['best_unseen_r50'].upper()}
- Highest Unseen R75: {answers['best_unseen_r75'].upper()}
- Lowest Seen Failure Rate: {answers['lowest_seen_failure'].upper()}
- Best Unseen Precision/FP rank trade-off: {answers['best_precision_fp_tradeoff'].upper()}
- Fastest model+NMS median latency: {answers['best_speed'].upper()}

Overall rank score (lower is better): {overall_scores}.

**BEST OVERALL EDGE OPERATOR: {overall.upper()}**

## Research interpretation

1. This is a controlled operator ablation, so performance differences are attributable primarily to the Shape-input representation under the fixed seed and pipeline.
2. Clean AP indicates whether the edge representation preserves ordinary detection quality rather than robustness alone.
3. Seen and Unseen AP separate adaptation to the training attack family from generalization to held-out patches.
4. R50 and R75 distinguish coarse recovery from stricter localization quality.
5. Failure Rate is interpreted jointly with AP because it measures losses among objects that were detectable in the paired clean image.
6. Precision and FP/image expose whether recall gains came from more false person detections.
7. The learned alpha values indicate how strongly optimization used each P3/P4/P5 Shape residual, not standalone causal importance.
8. Canny's thresholding and hysteresis produce sparse binary structure; Laplacian preserves dense second-derivative magnitude; Sobel preserves dense first-derivative magnitude.
9. The latency result includes edge extraction and NMS, making the final choice deployment-aware.
10. The overall verdict uses equal ranks over six accuracy/robustness criteria plus median latency; raw tables should remain the basis for alternative priorities.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps({
        "answers": answers, "overall_rank_score": overall_scores,
        "best_overall_edge_operator": overall,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "best_overall": overall, "report": str(OUT / "FINAL_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
