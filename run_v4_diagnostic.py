"""Bounded v4 universal-patch comparison on disjoint local COCO subsets."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
from .attacks import EOTConfig, PatchConfig, PatchGenerator
from .baseline import BaselineDetector
from .dataset import CocoPersonConfig, CocoPersonPatchDataset, coco_person_collate
from .evaluate_patch_attack import _coco_metrics, _run

def loader(dataset): return DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,collate_fn=coco_person_collate)

def evaluate(model, clean, patched, annotation, device, conf=.25):
    cp,cg,cm,ct=_run(model,loader(clean),device,conf); pp,pg,pm,pt=_run(model,loader(patched),device,conf)
    c=_coco_metrics(annotation,clean,cp,cg,cm); p=_coco_metrics(annotation,patched,pp,pg,pm)
    cc=[x for values in cm.values() for x in values]; pc=[x for values in pm.values() for x in values]; clean_hits=len(cc); lost=sum(max(0,len(cm.get(k,[]))-len(pm.get(k,[]))) for k in cm)
    cconf=sum(cc)/len(cc) if cc else 0.; pconf=sum(pc)/len(pc) if pc else 0.
    return {"clean":{**c,"mean_confidence":cconf},"patched":{**p,"mean_confidence":pconf},"drop":{"recall":c["recall50"]-p["recall50"],"ap50":c["ap50"]-p["ap50"],"confidence":cconf-pconf,"evaluation_failure_rate":lost/max(1,clean_hits)},"seconds":ct+pt}

def main():
    q=argparse.ArgumentParser(); q.add_argument("--checkpoint",type=Path,required=True); q.add_argument("--config",type=Path,default=Path("dataset_config.main80.yaml")); q.add_argument("--source-images",type=int,default=20); q.add_argument("--heldout-images",type=int,default=20); q.add_argument("--steps",type=int,default=30); q.add_argument("--batch-size",type=int,default=4); q.add_argument("--seed",type=int,default=2026); q.add_argument("--output",type=Path,default=Path("runs/v4_diagnostic")); a=q.parse_args(); root=Path(__file__).resolve().parent
    cfg=yaml.safe_load((a.config if a.config.is_absolute() else root/a.config).read_text(encoding="utf-8"));
    def paths(split):
        x=Path(split["images"]); y=Path(split["annotations"]); return (x if x.is_absolute() else root/x,y if y.is_absolute() else root/y)
    train_i,train_a=paths(cfg["train"]); val_i,val_a=paths(cfg["validation"]["clean"]); source=CocoPersonPatchDataset(CocoPersonConfig(train_i,train_a,max_images=a.source_images)); held=CocoPersonPatchDataset(CocoPersonConfig(val_i,val_a,max_images=a.heldout_images)); source_ids=[r[0]["id"] for r in source.records]; held_ids=[r[0]["id"] for r in held.records];
    if set(source_ids)&set(held_ids): raise RuntimeError("source and held-out image IDs overlap")
    samples=[source[i] for i in range(len(source))]; images=[s["img"].unsqueeze(0) for s in samples]; boxes=[s["bboxes"].unsqueeze(0) for s in samples]; classes=[s["cls"].squeeze(1).unsqueeze(0) for s in samples]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=BaselineDetector("yolo11n.pt").to(device); state=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(state.get("model",state),strict=False); model.eval(); a.output.mkdir(parents=True,exist_ok=True); results={}
    configs={"no_transform":EOTConfig(scale_min=1,scale_max=1,translate_fraction=0,rotation_degrees=0,perspective_jitter=0,brightness=0,contrast=0),"mild_transform":EOTConfig(scale_min=.9,scale_max=1.1,translate_fraction=.03,rotation_degrees=5,perspective_jitter=.02,brightness=.05,contrast=.05)}
    for name,eot in configs.items():
        gen=PatchGenerator(model,PatchConfig(steps=a.steps,seed=a.seed,objective="v3",eot=eot),device=device); res=gen.generate_many(images,boxes,classes,batch_size=a.batch_size); patch_path=a.output/f"patch_{name}.png"; gen.save(res,patch_path,{"objective":"v4_multi_image_v3","condition":name,"seed":a.seed,"steps":a.steps,"batch_size":a.batch_size,"source_ids":source_ids,"heldout_ids":held_ids,"eot":eot.__dict__})
        source_p=CocoPersonPatchDataset(CocoPersonConfig(train_i,train_a,patch_mode="seen",patch_dirs=(patch_path,),patch_probability=1,target_policy="person",max_images=a.source_images,seed=a.seed)); held_p=CocoPersonPatchDataset(CocoPersonConfig(val_i,val_a,patch_mode="seen",patch_dirs=(patch_path,),patch_probability=1,target_policy="person",max_images=a.heldout_images,seed=a.seed)); results[name]={"optimization":{"batch_loss_first":res.losses[0],"batch_loss_last":res.losses[-1],"universal_score_before":res.before_score,"universal_score_after":res.after_score,"gradient_norm_last":res.gradient_norms[-1]},"source":evaluate(model,source,source_p,train_a,device),"heldout":evaluate(model,held,held_p,val_a,device)}
    payload={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()),"settings":vars(a)|{"checkpoint":str(a.checkpoint)},"source_ids":source_ids,"heldout_ids":held_ids,"results":results,"metric_definition":"Recall@0.5 uses NMS person predictions with confidence>=0.25, greedy one-prediction-per-GT matching; failure rate is clean-matched GT lost after perturbation."}; (a.output/"results.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); print(json.dumps(results,indent=2))
if __name__=="__main__": main()
