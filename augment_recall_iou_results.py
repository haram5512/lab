"""Create recall/IoU comparison artifacts from completed final evaluation JSONs."""
import csv, json
from pathlib import Path

root=Path(__file__).resolve().parent
out=root/"runs"/"main_80class"/"proposed_rgb_shape_v1_final"
names=["pretrained","adv_best_seen","proposed_best_clean","proposed_best_seen"]
modes=["clean","seen","unseen"]
results={(n,m):json.loads((out/f"{n}_{m}.json").read_text(encoding="utf-8")) for n in names for m in modes}
rows=[]
for n in names:
    for m in modes:
        x=results[(n,m)]
        rows.append({"model":n,"condition":m,"person_ap":x["person_ap"],"person_ap50":x["person_ap50"],"person_ap75":x["person_ap75"],"recall_iou50":x["person_recall_iou50_conf025"],"recall_iou75":x["person_recall_iou75_conf025"],"mean_matched_iou":x["mean_matched_iou"],"median_matched_iou":x["median_matched_iou"],"std_matched_iou":x["std_matched_iou"],"matched_count":x["matched_count"],"mean_gt_iou_with_misses":x["mean_gt_iou_with_misses"],"failure_rate":x["failure_rate"]})
comparison={}
for n in names[1:]:
    c,s,u=results[(n,"clean")],results[(n,"seen")],results[(n,"unseen")]
    r={}
    for metric in ("person_recall_iou50_conf025","person_recall_iou75_conf025","mean_matched_iou","mean_gt_iou_with_misses"):
        r[f"{metric}_clean_minus_seen"]=c[metric]-s[metric]
        r[f"{metric}_clean_minus_unseen"]=c[metric]-u[metric]
    for ref_name in ("pretrained","adv_best_seen"):
        ref={m:results[(ref_name,m)] for m in modes}
        for mode in ("clean","seen","unseen"):
            for metric in ("person_ap","person_ap50","person_ap75","person_recall_iou50_conf025","person_recall_iou75_conf025","mean_matched_iou","mean_gt_iou_with_misses"):
                r[f"{mode}_{metric}_vs_{ref_name}"]=results[(n,mode)][metric]-ref[mode][metric]
    comparison[n]=r
(out/"comparison_with_recall_iou.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
(out/"iou_diagnostics.json").write_text(json.dumps({"rows":rows,"comparison":comparison},indent=2),encoding="utf-8")
with (out/"comparison_with_recall_iou.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
summary=json.loads((out/"tradeoff_summary.json").read_text(encoding="utf-8"))
summary["comparison_with_recall_iou"]=comparison
summary["final_verdict"]="NO-GO"
summary["verdict_basis"]="Proposed v1 does not improve Recall@0.50/0.75 or IoU over conventional adversarial training on seen/unseen; unseen AP and GT-inclusive IoU are lower, while clean AP is also lower."
(out/"tradeoff_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({"rows":len(rows),"final_verdict":"NO-GO"},indent=2))
