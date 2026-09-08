"""Full Clean/Seen/Unseen evaluation for Proposed v3 checkpoints and integration."""
from __future__ import annotations
import contextlib, csv, json, os
from copy import deepcopy
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent/"runs"/".ultralytics"))
from .coco_eval_core import evaluate_model
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config

ROOT=Path(__file__).resolve().parent; BASE=ROOT/"yolo11n.pt"; OUT=ROOT.parent/"runs"/"main_80class"/"proposed_rgb_shape_v3_final_eval"; V3=ROOT.parent/"runs"/"main_80class"/"proposed_rgb_shape_v3_full"

def load(path,device):
    model=ProposedRGBShapeV3(ProposedV3Config(weights=str(BASE))); model.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["model"],strict=True); return model.to(device).eval()

def failure(clean,attack):
    clean_status=clean.pop("person_match_status"); attack_status=attack.pop("person_match_status"); successes=failures=0
    for image_id,flags in clean_status.items():
        for clean_ok,attack_ok in zip(flags,attack_status[image_id]): successes+=int(clean_ok); failures+=int(clean_ok and not attack_ok)
    return failures/successes if successes else None

def main():
    OUT.mkdir(parents=True,exist_ok=True); config=yaml.safe_load((ROOT/"dataset_config.main80.yaml").read_text(encoding="utf-8")); val=config["validation"]["clean"]; images=(ROOT/Path(val["images"])).resolve(); annotations=(ROOT/Path(val["annotations"])).resolve(); payload=json.loads(annotations.read_text(encoding="utf-8")); cats=tuple(sorted(int(x["id"]) for x in payload["categories"])); common=dict(images_dir=images,annotations_file=annotations,image_size=640,max_images=None,target_policy="person",required_category_ids=(),category_ids=cats,seed=7,position_jitter=.015,placement_mode="torso",torso_relative_y=.38,patch_scale_min=.95,patch_scale_max=1.05,patch_rotation_degrees=3.,patch_brightness_jitter=.03,patch_perspective_jitter=.005); patches={"seen":(ROOT/"runs"/"v5_diagnostic"/"patch_V5-C.png",),"unseen":(ROOT/"artifacts"/"adversarial_patches"/"unseen",)}
    datasets={"clean":Coco80DetectionDataset(CocoPersonConfig(**common,patch_mode="clean",patch_probability=0.,patch_dirs=())),"seen":Coco80DetectionDataset(CocoPersonConfig(**common,patch_mode="seen",patch_probability=1.,patch_dirs=patches["seen"])),"unseen":Coco80DetectionDataset(CocoPersonConfig(**common,patch_mode="unseen",patch_probability=1.,patch_dirs=patches["unseen"]))}; loaders={m:DataLoader(d,batch_size=16,shuffle=False,collate_fn=coco_person_collate,num_workers=0) for m,d in datasets.items()}; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); results={}
    for ckpt_name in ("best_clean_map","best_seen_map"):
        path=V3/f"{ckpt_name}.pt"; model=load(path,device); values={}
        for mode in ("clean","seen","unseen"):
            with (OUT/f"v3_{ckpt_name}_{mode}.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log): values[mode]=evaluate_model(model,loaders[mode],annotations,datasets[mode],device,640,mode,confidence=.001,iou_threshold=.7)
        for mode in ("seen","unseen"): values[mode]["failure_rate"]=failure(deepcopy(values["clean"]),values[mode])
        values["clean"].pop("person_match_status")
        results[ckpt_name]=values
        for mode,metric in values.items(): (OUT/f"v3_{ckpt_name}_{mode}.json").write_text(json.dumps(metric,indent=2),encoding="utf-8")
        del model; torch.cuda.empty_cache()
    rows=[]
    for ckpt,values in results.items():
        for mode,m in values.items(): rows.append({"model":"proposed_v3","checkpoint":ckpt,"condition":mode,**{k:m.get(k) for k in ("person_ap","person_ap50","person_ap75","person_ar100","person_recall_iou50_conf025","person_recall_iou75_conf025","precision_iou50_conf025","f1_iou50_conf025","mean_matched_iou","mean_gt_iou_with_misses","fp_per_image_conf025","failure_rate")}})
    (OUT/"v3_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8");
    with (OUT/"v3_metrics.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    old_csv=ROOT.parent/"runs"/"main_80class"/"final_model_comparison"/"all_models_metrics.csv"; old_rows=list(csv.DictReader(old_csv.open(encoding="utf-8"))); combined=old_rows+rows; fields=list(rows[0]);
    with (ROOT.parent/"runs"/"main_80class"/"final_model_comparison"/"all_models_metrics_with_v3.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(combined)
    old_json=json.loads((ROOT.parent/"runs"/"main_80class"/"final_model_comparison"/"all_models_metrics.json").read_text(encoding="utf-8")); old_json["proposed_v3"]={k:v for k,v in results.items()}; (ROOT.parent/"runs"/"main_80class"/"final_model_comparison"/"all_models_metrics_with_v3.json").write_text(json.dumps(old_json,indent=2),encoding="utf-8")
    adv=next(r for r in old_rows if r["model"]=="conventional_adv" and r["checkpoint"]=="best_seen_map" and r["condition"]=="unseen"); v2=next(r for r in old_rows if r["model"]=="proposed_v2" and r["checkpoint"]=="best_seen_map" and r["condition"]=="unseen"); comparisons=[]
    for ckpt in results:
        for mode in ("clean","seen","unseen"):
            v3=results[ckpt][mode]
            for metric in ("person_ap","person_recall_iou50_conf025","person_recall_iou75_conf025","precision_iou50_conf025","f1_iou50_conf025","mean_gt_iou_with_misses","failure_rate"):
                a=float(adv[metric.replace("person_recall_iou50_conf025","person_recall_iou50_conf025").replace("person_recall_iou75_conf025","person_recall_iou75_conf025")]) if mode=="unseen" and metric in adv else None
                comparisons.append({"checkpoint":ckpt,"condition":mode,"metric":metric,"v3":v3.get(metric),"adv_unseen_reference":a})
    with (ROOT.parent/"runs"/"main_80class"/"final_model_comparison"/"v2_vs_v3_vs_adv.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    report="# Proposed v3 Final Evaluation\n\nProposed v3 checkpoints were evaluated on the full COCO val2017 person-containing set using the same corrected evaluator as the previous models. See `v3_metrics.csv` and `all_models_metrics_with_v3.csv` for complete metrics.\n"; (OUT/"FINAL_REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps({"output":str(OUT),"conditions":6,"combined_csv":str(ROOT.parent/"runs"/"main_80class"/"final_model_comparison"/"all_models_metrics_with_v3.csv")},indent=2))
if __name__=="__main__": main()
