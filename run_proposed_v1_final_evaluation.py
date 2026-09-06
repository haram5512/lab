"""Full COCO clean/seen/unseen evaluation for pretrained, conventional Adv, and Proposed v1."""
from __future__ import annotations
import contextlib, hashlib, json, subprocess, time
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
from .baseline import BaselineDetector
from .coco_eval_core import evaluate_model
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape import ProposedRGBShapeV1, ProposedV1Config

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "main_80class" / "proposed_rgb_shape_v1_final"
BASE = ROOT / "yolo11n.pt"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def load(kind: str, path: Path, device: torch.device):
    if kind == "proposed":
        model = ProposedRGBShapeV1(ProposedV1Config(weights=str(BASE)))
    else:
        model = BaselineDetector(str(BASE))
    if path.resolve() != BASE.resolve():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()

def latency(model, device):
    x = torch.rand(1, 3, 640, 640, device=device)
    with torch.inference_mode():
        for _ in range(10): model.predict(x, confidence=.25, iou=.7, classes=None)
        torch.cuda.synchronize(); values=[]
        for _ in range(30):
            start=time.perf_counter(); model.predict(x, confidence=.25, iou=.7, classes=None); torch.cuda.synchronize(); values.append((time.perf_counter()-start)*1000)
    t=torch.tensor(values); med=float(t.median())
    return {"median_ms":med,"mean_ms":float(t.mean()),"p95_ms":float(torch.quantile(t,.95)),"fps":1000/med}

def main():
    torch.manual_seed(42)
    root=ROOT; OUT.mkdir(parents=True, exist_ok=True); logs=OUT/"logs"; logs.mkdir(exist_ok=True)
    cfg=yaml.safe_load((root/"dataset_config.main80.yaml").read_text(encoding="utf-8")); val=cfg["validation"]["clean"]
    images=(root/Path(val["images"])).resolve(); annotations=(root/Path(val["annotations"])).resolve()
    ann=json.loads(annotations.read_text(encoding="utf-8")); all_ids=tuple(sorted(int(x["id"]) for x in ann["categories"]))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv=root/"runs"/"main_80class"/"v5c_adversarial_training_corrected"; proposed=root/"runs"/"main_80class"/"proposed_rgb_shape_v1_full"
    models={"pretrained":("baseline",BASE),"adv_best_seen":("baseline",adv/"best_seen_map.pt"),"proposed_best_clean":("proposed",proposed/"best_clean_map.pt"),"proposed_best_seen":("proposed",proposed/"best_seen_map.pt")}
    for name,(_,path) in models.items():
        if not path.is_file() or path.stat().st_size==0: raise FileNotFoundError(path)
    patch_sources={"seen":(root/"runs"/"v5_diagnostic"/"patch_V5-C.png",),"unseen":(root/"artifacts"/"adversarial_patches"/"unseen",)}
    common={"images_dir":images,"annotations_file":annotations,"image_size":640,"max_images":None,"target_policy":"person","required_category_ids":(),"position_jitter":.015,"placement_mode":"torso","torso_relative_y":.38,"patch_scale_min":.95,"patch_scale_max":1.05,"patch_rotation_degrees":3.,"patch_brightness_jitter":.03,"patch_perspective_jitter":.005,"category_ids":all_ids}
    results={name:{} for name in models}; latencies={}
    for name,(kind,path) in models.items():
        model=load(kind,path,device)
        latencies[name]={"checkpoint":str(path),"forward_plus_nms":latency(model,device),"parameters":sum(p.numel() for p in model.parameters())}
        for mode in ("clean","seen","unseen"):
            if mode=="clean": dc=CocoPersonConfig(**common,patch_mode="clean",patch_probability=0.,patch_dirs=())
            else: dc=CocoPersonConfig(**common,patch_mode=mode,patch_probability=1.,patch_dirs=patch_sources[mode])
            ds=Coco80DetectionDataset(dc); loader=DataLoader(ds,batch_size=16,shuffle=False,collate_fn=coco_person_collate,num_workers=0)
            with (logs/f"{name}_{mode}.log").open("w",encoding="utf-8") as log, contextlib.redirect_stdout(log):
                metrics=evaluate_model(model,loader,annotations,ds,device,640,mode,confidence=.001,iou_threshold=.7)
            metrics.update({"model":name,"checkpoint":str(path),"mode":mode,"images":len(ds),"person_gt_expected":11004})
            results[name][mode]=metrics
            (OUT/f"{name}_{mode}.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        del model; torch.cuda.empty_cache()
    for mode in ("clean","seen","unseen"):
        (OUT/f"{mode}_results.json").write_text(json.dumps({n:results[n][mode] for n in results},indent=2),encoding="utf-8")
    ref=results["pretrained"]
    comparison={}
    for name in ("adv_best_seen","proposed_best_clean","proposed_best_seen"):
        clean,seen,unseen=results[name]["clean"],results[name]["seen"],results[name]["unseen"]
        comparison[name]={"clean_degradation":{"person_ap":ref["clean"]["person_ap"]-clean["person_ap"],"person_ap50":ref["clean"]["person_ap50"]-clean["person_ap50"],"person_ap75":ref["clean"]["person_ap75"]-clean["person_ap75"],"person_recall":ref["clean"]["person_recall"]-clean["person_recall"],"person_ar100":ref["clean"]["person_ar100"]-clean["person_ar100"]},"seen_vs_pretrained":{"person_ap":seen["person_ap"]-ref["seen"]["person_ap"],"person_ap50":seen["person_ap50"]-ref["seen"]["person_ap50"],"person_recall":seen["person_recall"]-ref["seen"]["person_recall"]},"unseen_vs_pretrained":{"person_ap":unseen["person_ap"]-ref["unseen"]["person_ap"],"person_ap50":unseen["person_ap50"]-ref["unseen"]["person_ap50"],"person_recall":unseen["person_recall"]-ref["unseen"]["person_recall"]},"seen_vs_adv":{"person_ap":seen["person_ap"]-results["adv_best_seen"]["seen"]["person_ap"],"person_recall":seen["person_recall"]-results["adv_best_seen"]["seen"]["person_recall"]},"unseen_vs_adv":{"person_ap":unseen["person_ap"]-results["adv_best_seen"]["unseen"]["person_ap"],"person_recall":unseen["person_recall"]-results["adv_best_seen"]["unseen"]["person_recall"]},"generalization_gap":{"seen_minus_unseen_ap":seen["person_ap"]-unseen["person_ap"],"seen_minus_unseen_recall":seen["person_recall"]-unseen["person_recall"]}}
    (OUT/"comparison.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8"); (OUT/"latency_results.json").write_text(json.dumps(latencies,indent=2),encoding="utf-8")
    commit=subprocess.check_output(["git","-c",f"safe.directory={root}","rev-parse","HEAD"],cwd=root,text=True).strip()
    metadata={"git_commit":commit,"dataset":"COCO val2017 full image set","person_gt_expected":11004,"image_size":640,"confidence":.001,"nms_iou":.7,"models":{n:{"kind":k,"path":str(p),"sha256":sha256(p)} for n,(k,p) in models.items()},"patches":{m:[str(p) for p in ps] for m,ps in patch_sources.items()},"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","torch":torch.__version__,"results":str(OUT)}
    (OUT/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (OUT/"tradeoff_summary.json").write_text(json.dumps({"pretrained_reference":{m:{k:ref[m].get(k) for k in ("person_ap","person_ap50","person_ap75","person_ar100","person_recall") } for m in ref},"comparison":comparison,"recommendation":"select checkpoint after clean_seen_unseen_tradeoff_review"},indent=2),encoding="utf-8")
    print(json.dumps({"output":str(OUT),"models":list(models),"modes":["clean","seen","unseen"]},indent=2))

if __name__=="__main__": main()
