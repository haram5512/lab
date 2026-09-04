"""Small local diagnostic linking raw GT-matched scores to final person detections."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import torch, yaml
from .attacks import EOTConfig, PatchConfig, PatchGenerator
from .baseline import BaselineDetector
from .dataset import CocoPersonConfig, CocoPersonPatchDataset

def xywh_xyxy(box):
    x,y,w,h=box; return torch.tensor([x-w/2,y-h/2,x+w/2,y+h/2],device=box.device)

def iou(one,many):
    lt=torch.maximum(one[:2],many[:,:2]); rb=torch.minimum(one[2:],many[:,2:]); wh=(rb-lt).clamp(min=0); inter=wh[:,0]*wh[:,1]
    return inter/(((one[2]-one[0])*(one[3]-one[1]))+(many[:,2]-many[:,0])*(many[:,3]-many[:,1])-inter).clamp(min=1e-9)

def target_detection(model,image,gt,conf=.25):
    det=model.predict(image,confidence=conf,iou=.7,classes=[0])[0]
    if not len(det): return 0.0
    overlap=iou(xywh_xyxy(gt.reshape(-1,4)[0]*image.shape[-1]),det[:,:4])
    valid=det[overlap>=.5]
    return float(valid[:,4].max()) if len(valid) else 0.0

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--config",type=Path,default=Path("dataset_config.main80.yaml")); p.add_argument("--images",type=int,default=5); p.add_argument("--steps",type=int,default=50); p.add_argument("--output",type=Path,default=Path("runs/v3_diagnostic")); a=p.parse_args(); root=Path(__file__).resolve().parent
    cfg=yaml.safe_load((a.config if a.config.is_absolute() else root/a.config).read_text(encoding="utf-8")); split=cfg["train"]; ip=Path(split["images"]); ap=Path(split["annotations"]); ip=ip if ip.is_absolute() else root/ip; ap=ap if ap.is_absolute() else root/ap
    ds=CocoPersonPatchDataset(CocoPersonConfig(ip,ap,image_size=640,max_images=a.images)); samples=[ds[i] for i in range(len(ds))]; images=[s["img"].unsqueeze(0) for s in samples]; boxes=[s["bboxes"][:1].unsqueeze(0) for s in samples]; classes=[torch.zeros((1,1)) for _ in samples]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=BaselineDetector("yolo11n.pt").to(device); state=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(state.get("model",state),strict=False); model.eval()
    eot=EOTConfig(scale_min=.9,scale_max=1.1,translate_fraction=.03,rotation_degrees=5,brightness=.05,contrast=.05,perspective_jitter=.02)
    results={}; generated={}
    for objective in ("v2","v3"):
        gen=PatchGenerator(model,PatchConfig(steps=a.steps,seed=123,objective=objective,eot=eot),device=device); res=gen.generate_many(images,boxes,classes,batch_size=len(images)); generated[objective]=(gen,res)
        gen.save(res,a.output/f"patch_{objective}.png",{"objective":objective,"seed":123,"steps":a.steps,"source_images":len(images),"checkpoint":str(a.checkpoint)})
        scores=[]
        for image,box in zip(images,boxes): scores.append(target_detection(model,gen._place(res.patch.to(device),image.to(device),box.to(device)[0,0]),box.to(device)))
        results[objective]={"loss_first":res.losses[0],"loss_last":res.losses[-1],"gradient_norm_first":res.gradient_norms[0],"gradient_norm_last":res.gradient_norms[-1],"matched_detection_scores":scores,"detected":sum(x>0 for x in scores)}
    clean_scores=[target_detection(model,i.to(device),b.to(device)) for i,b in zip(images,boxes)]
    diagnostics=[]
    with torch.no_grad():
        for sample,image,gt in zip(samples[:3],images[:3],boxes[:3]):
            raw=model(image.to(device)); pred=(raw[0] if isinstance(raw,(tuple,list)) else raw).transpose(1,2)[0]; cxywh=pred[:,:4]; cboxes=torch.stack([cxywh[:,0]-cxywh[:,2]/2,cxywh[:,1]-cxywh[:,3]/2,cxywh[:,0]+cxywh[:,2]/2,cxywh[:,1]+cxywh[:,3]/2],1); g=xywh_xyxy(gt.to(device)[0,0]*640); overlaps=iou(g,cboxes); person=pred[:,4]; top=torch.topk(person*overlaps.clamp(min=0),5).indices; diagnostics.append({"image_id":sample["image_id"],"gt_xyxy":g.cpu().tolist(),"raw_shape":list((raw[0] if isinstance(raw,(tuple,list)) else raw).shape),"top_candidates":[{"box":cboxes[j].cpu().tolist(),"person_score":float(person[j]),"iou":float(overlaps[j]),"v2_included":bool((cxywh[j,0]>=g[0])&(cxywh[j,0]<=g[2])&(cxywh[j,1]>=g[1])&(cxywh[j,1]<=g[3])),"v3_eligible":bool(overlaps[j]>=.1)} for j in top]})
    payload={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"settings":{"images":len(images),"steps":a.steps,"seed":123},"clean":{"matched_detection_scores":clean_scores,"detected":sum(x>0 for x in clean_scores)},"objectives":results,"candidate_diagnostics":diagnostics}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/"diagnostic.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps({"clean":payload["clean"],"objectives":results},indent=2))
if __name__=="__main__": main()
