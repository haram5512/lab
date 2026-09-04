"""Held-out COCO person attack evaluation (clean versus one patched tensor)."""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .baseline import BaselineDetector
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .metrics import inverse_letterbox_xyxy

def _iou(a, b):
    lt=torch.maximum(a[:2],b[:,:2]); rb=torch.minimum(a[2:],b[:,2:]); wh=(rb-lt).clamp(min=0)
    inter=wh[:,0]*wh[:,1]; aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/(aa+ab-inter).clamp(min=1e-9)

def _run(model, loader, device, conf):
    out={}; matched={}; gts={}; start=time.perf_counter(); model.eval()
    with torch.no_grad():
        for batch in loader:
            dets=model.predict(batch["img"].to(device), confidence=0.001, iou=0.7, classes=[0])
            for i,d in enumerate(dets):
                boxes=inverse_letterbox_xyxy(d[:,:4].cpu(),batch["original_size"][i],batch["letterbox_scale"][i],batch["letterbox_pad"][i]) if len(d) else torch.empty((0,4))
                gt=[]
                for box,cls in zip(batch["bboxes"][batch["batch_idx"]==i],batch["cls"][batch["batch_idx"]==i]):
                    if int(cls.item())==0:
                        x,y,w,h=box; gt.append(torch.tensor([(x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640]))
                gts[int(batch["image_id"][i])]=gt; out[int(batch["image_id"][i])]=(boxes,d[:,4].cpu() if len(d) else torch.empty(0))
                used=set(); vals=[]
                for target in gt:
                    candidates=[j for j in range(len(boxes)) if j not in used and float(d[j,4])>=conf]
                    if candidates:
                        scores=_iou(target,boxes[candidates]); k=int(scores.argmax());
                        if float(scores[k])>=0.5: used.add(candidates[k]); vals.append(float(d[candidates[k],4]))
                matched[int(batch["image_id"][i])]=vals
    return out,gts,matched,time.perf_counter()-start

def _coco_metrics(annotations, dataset, preds, gts):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco=COCO(str(annotations)); rows=[]
    for image_id,(boxes,scores) in preds.items():
        for box,score in zip(boxes,scores):
            rows.append({"image_id":image_id,"category_id":1,"bbox":[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])],"score":float(score)})
    result=coco.loadRes(rows) if rows else coco.loadRes([]); ev=COCOeval(coco,result,"bbox"); ev.params.imgIds=list(preds); ev.evaluate(); ev.accumulate(); ev.summarize()
    precision=ev.eval["precision"][:,:,0,0,-1]; valid=precision[precision>-1]
    p50=precision[0]; p75=precision[5]; p50=p50[p50>-1]; p75=p75[p75>-1]
    gt_count=sum(len(v) for v in gts.values()); tp=sum(len(v) for v in gts.values() if v)
    return {"ap":float(valid.mean()) if valid.size else 0.0,"ap50":float(p50.mean()) if p50.size else 0.0,"ap75":float(p75.mean()) if p75.size else 0.0,"recall50":tp/gt_count if gt_count else 0.0,"gt_person_count":gt_count}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--patch",type=Path,required=True); p.add_argument("--config",type=Path,default=Path("dataset_config.main80.yaml")); p.add_argument("--max-images",type=int,default=100); p.add_argument("--image-size",type=int,default=640); p.add_argument("--mode",choices=("deterministic","eot"),default="deterministic"); p.add_argument("--seed",type=int,default=42); p.add_argument("--conf-threshold",type=float,default=0.25); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); root=Path(__file__).resolve().parent
    if not args.checkpoint.exists(): raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.patch.exists(): raise SystemExit(f"patch not found: {args.patch}")
    cfg=__import__("yaml").safe_load((root/args.config if not args.config.is_absolute() else args.config).read_text(encoding="utf-8")); split=cfg["validation"]["clean"]; images=Path(split["images"]); ann=Path(split["annotations"]); images=images if images.is_absolute() else root/images; ann=ann if ann.is_absolute() else root/ann; limit=None if args.max_images<=0 else args.max_images
    clean=Coco80DetectionDataset(CocoPersonConfig(images,ann,image_size=args.image_size,patch_mode="clean",target_policy="person",max_images=limit)); patched=Coco80DetectionDataset(CocoPersonConfig(images,ann,image_size=args.image_size,patch_mode="seen",patch_dirs=(args.patch,),patch_probability=1.0,target_policy="person",max_images=limit));
    if [r[0]["id"] for r in clean.records] != [r[0]["id"] for r in patched.records]: raise RuntimeError("clean/patched image subsets differ")
    loader=lambda d: DataLoader(d,batch_size=8,shuffle=False,collate_fn=coco_person_collate,num_workers=0); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=BaselineDetector("yolo11n.pt").to(device); state=torch.load(args.checkpoint,map_location=device,weights_only=False); model.load_state_dict(state.get("model",state),strict=False)
    cp,cg,cm,ct=_run(model,loader(clean),device,args.conf_threshold); pp,pg,pm,pt=_run(model,loader(patched),device,args.conf_threshold); c=_coco_metrics(ann,clean,cp,cg); q=_coco_metrics(ann,patched,pp,pg); clean_conf=[x for v in cm.values() for x in v]; patch_conf=[x for v in pm.values() for x in v]; base=sum(clean_conf)/len(clean_conf) if clean_conf else 0.; adv=sum(patch_conf)/len(patch_conf) if patch_conf else 0.; clean_hits=sum(len(v) for v in cm.values()); lost=sum(max(0,len(cm.get(k,[]))-len(pm.get(k,[]))) for k in cm); asr=lost/max(1,clean_hits); result={"checkpoint":str(args.checkpoint.resolve()),"patch":str(args.patch.resolve()),"mode":args.mode,"image_count":len(clean),"image_ids":[int(r[0]["id"]) for r in clean.records],"clean":{**c,"mean_matched_confidence":base},"patched":{**q,"mean_matched_confidence":adv},"attack":{"ap_drop":c["ap"]-q["ap"],"ap50_drop":c["ap50"]-q["ap50"],"recall_drop":c["recall50"]-q["recall50"],"relative_recall_drop":(c["recall50"]-q["recall50"])/c["recall50"] if c["recall50"] else 0.0,"mean_confidence_drop":base-adv,"asr":asr},"runtime":{"clean_seconds":ct,"patched_seconds":pt},"git":{"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()}}
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"attack_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result["attack"],indent=2))
if __name__=="__main__": main()
