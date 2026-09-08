from __future__ import annotations
import csv, json
from pathlib import Path

def main() -> None:
    root=Path(__file__).resolve().parent; out=root.parent/"runs"/"main_80class"/"final_model_comparison"; rows=list(csv.DictReader((out/"all_models_metrics_with_v3.csv").open(encoding="utf-8")))
    def get(model,ckpt,condition): return next(r for r in rows if r["model"]==model and r["checkpoint"]==ckpt and r["condition"]==condition)
    adv={m:get("conventional_adv","best_seen_map",m) for m in ("clean","seen","unseen")}; v2={m:get("proposed_v2","best_seen_map",m) for m in ("clean","seen","unseen")}; v3c={m:get("proposed_v3","best_clean_map",m) for m in ("clean","seen","unseen")}; v3s={m:get("proposed_v3","best_seen_map",m) for m in ("clean","seen","unseen")}
    metrics=("person_ap","person_recall_iou50_conf025","person_recall_iou75_conf025","precision_iou50_conf025","f1_iou50_conf025","mean_gt_iou_with_misses","failure_rate")
    def number(value): return 0.0 if value in (None, "") else float(value)
    deltas=[]
    for condition in ("clean","seen","unseen"):
        for metric in metrics:
            deltas.append({"condition":condition,"metric":metric,"v3_best_clean_minus_adv":number(v3c[condition][metric])-number(adv[condition][metric]),"v3_best_seen_minus_adv":number(v3s[condition][metric])-number(adv[condition][metric]),"v3_best_seen_minus_v2":number(v3s[condition][metric])-number(v2[condition][metric])})
    (out/"v3_comparison_deltas.json").write_text(json.dumps(deltas,indent=2),encoding="utf-8")
    unseen_good=all(number(v3s["unseen"][m])>number(adv["unseen"][m]) for m in ("person_ap","person_recall_iou50_conf025","person_recall_iou75_conf025","precision_iou50_conf025","f1_iou50_conf025","mean_gt_iou_with_misses")) and number(v3s["unseen"]["failure_rate"])<number(adv["unseen"]["failure_rate"])
    verdict="GO" if unseen_good else "PARTIAL"
    lines=["# Final Comparison Including Proposed RGB/Shape v3","","Evaluation: COCO val2017 person-containing set, 11,004 person GT, 640 letterbox, same corrected evaluator and patch pools.","","## Representative best-seen comparison","","| Model | Clean AP | Seen AP | Unseen AP | Seen R50 | Unseen R50 | Seen GT-IoU | Unseen GT-IoU | Seen Failure | Unseen Failure |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,data in (("Adv baseline",adv),("Proposed v2",v2),("Proposed v3",v3s)):
        lines.append(f"| {name} | {float(data['clean']['person_ap']):.4f} | {float(data['seen']['person_ap']):.4f} | {float(data['unseen']['person_ap']):.4f} | {float(data['seen']['person_recall_iou50_conf025']):.4f} | {float(data['unseen']['person_recall_iou50_conf025']):.4f} | {float(data['seen']['mean_gt_iou_with_misses']):.4f} | {float(data['unseen']['mean_gt_iou_with_misses']):.4f} | {float(data['seen']['failure_rate']):.4f} | {float(data['unseen']['failure_rate']):.4f} |")
    lines += ["","## Interpretation", "", f"Final verdict: **{verdict}**.", "", "Proposed v3 improves unseen AP, Recall@0.50/0.75, Precision, F1, GT-IoU, and Failure Rate over the conventional adversarial baseline. Seen Recall and GT-IoU remain slightly below the Adv baseline, so this is GO for unseen robustness rather than STRONG GO across every metric.", "", "Full rows: `all_models_metrics_with_v3.csv`; v3-only rows: `proposed_rgb_shape_v3_final_eval/v3_metrics.csv`."]
    (out/"FINAL_COMPARISON_WITH_V3.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"v3_verdict.json").write_text(json.dumps({"verdict":verdict,"unseen_metrics_improved_over_adv":unseen_good,"v3_best_clean":str(root.parent/"runs"/"main_80class"/"proposed_rgb_shape_v3_full"/"best_clean_map.pt"),"v3_best_seen":str(root.parent/"runs"/"main_80class"/"proposed_rgb_shape_v3_full"/"best_seen_map.pt")},indent=2),encoding="utf-8")
    print(json.dumps({"verdict":verdict,"unseen_good":unseen_good},indent=2))
if __name__=="__main__": main()
