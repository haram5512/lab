"""Final bounded universal-patch transfer study: budget, diversity, torso placement."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import torch, yaml
from .attacks import EOTConfig, PatchConfig, PatchGenerator
from .baseline import BaselineDetector
from .dataset import CocoPersonConfig, CocoPersonPatchDataset
from .run_v4_diagnostic import evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--config",type=Path,default=Path("dataset_config.main80.yaml")); p.add_argument("--source-images",type=int,default=100); p.add_argument("--heldout-images",type=int,default=30); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--seed",type=int,default=2027); p.add_argument("--output",type=Path,default=Path("runs/v5_diagnostic")); a=p.parse_args(); root=Path(__file__).resolve().parent; a.output.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((a.config if a.config.is_absolute() else root/a.config).read_text(encoding="utf-8"));
    def paths(split):
        i=Path(split["images"]); n=Path(split["annotations"]); return (i if i.is_absolute() else root/i,n if n.is_absolute() else root/n)
    ti,ta=paths(cfg["train"]); vi,va=paths(cfg["validation"]["clean"]); source=CocoPersonPatchDataset(CocoPersonConfig(ti,ta,max_images=a.source_images)); held=CocoPersonPatchDataset(CocoPersonConfig(vi,va,max_images=a.heldout_images)); source_ids=[r[0]["id"] for r in source.records]; held_ids=[r[0]["id"] for r in held.records]
    if set(source_ids)&set(held_ids): raise RuntimeError("source/held-out overlap")
    samples=[source[i] for i in range(len(source))]; images=[s["img"].unsqueeze(0) for s in samples]; boxes=[s["bboxes"].unsqueeze(0) for s in samples]; classes=[s["cls"].squeeze(1).unsqueeze(0) for s in samples]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=BaselineDetector("yolo11n.pt").to(device); state=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(state.get("model",state),strict=False); model.eval()
    identity=EOTConfig(scale_min=1,scale_max=1,translate_fraction=0,rotation_degrees=0,perspective_jitter=0,brightness=0,contrast=0); mild=EOTConfig(scale_min=.95,scale_max=1.05,translate_fraction=.015,rotation_degrees=3,perspective_jitter=.005,brightness=.03,contrast=.03); convergence={}; patches={}
    for steps in (20,60,120):
        gen=PatchGenerator(model,PatchConfig(steps=steps,seed=a.seed,objective="v3",placement_mode="center",eot=identity),device=device); res=gen.generate_many(images,boxes,classes,batch_size=a.batch_size); path=a.output/f"patch_budget_{steps}.png"; gen.save(res,path,{"condition":"budget","steps":steps,"source_ids":source_ids,"heldout_ids":held_ids}); convergence[str(steps)]={"score_before":res.before_score,"score_after":res.after_score,"gradient_norm_last":res.gradient_norms[-1]}; patches["V5-A"]=path
    conditions={"V5-A":("center",identity,patches["V5-A"]),"V5-B":("torso",identity,None),"V5-C":("torso",mild,None)}; results={}
    for name,(placement,eot,path) in conditions.items():
        if path is None:
            gen=PatchGenerator(model,PatchConfig(steps=120,seed=a.seed,objective="v3",placement_mode=placement,eot=eot),device=device); res=gen.generate_many(images,boxes,classes,batch_size=a.batch_size); path=a.output/f"patch_{name}.png"; gen.save(res,path,{"condition":name,"steps":120,"placement":placement,"eot":eot.__dict__,"source_ids":source_ids,"heldout_ids":held_ids})
        def patched(i,n,count):
            kw={"patch_mode":"seen","patch_dirs":(path,),"patch_probability":1,"target_policy":"person","max_images":count,"seed":a.seed,"placement_mode":placement,"position_jitter":0,"patch_scale_min":1,"patch_scale_max":1,"patch_rotation_degrees":0,"patch_brightness_jitter":0,"patch_perspective_jitter":0}
            if name=="V5-C": kw.update(position_jitter=.015,patch_scale_min=.95,patch_scale_max=1.05,patch_rotation_degrees=3,patch_brightness_jitter=.03,patch_perspective_jitter=.005)
            return CocoPersonPatchDataset(CocoPersonConfig(i,n,**kw))
        results[name]={"source":evaluate(model,source,patched(ti,ta,a.source_images),ta,device),"heldout":evaluate(model,held,patched(vi,va,a.heldout_images),va,device)}; results[name]["gap"]={k:results[name]["source"]["drop"][k]-results[name]["heldout"]["drop"][k] for k in ("recall","ap50","confidence")}
    meta={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()),"checkpoint":str(a.checkpoint),"seed":a.seed,"source_ids":source_ids,"heldout_ids":held_ids,"image_size":640,"batch_size":a.batch_size,"optimizer":"Adam","learning_rate":.03,"candidate_iou_threshold":.1,"top_k":10,"confidence_threshold":.25,"evaluation_iou_threshold":.5,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","torch":torch.__version__,"cuda":torch.version.cuda,"convergence":convergence,"results":results}; (a.output/"results.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps({"convergence":convergence,"results":results},indent=2))
if __name__=="__main__": main()
