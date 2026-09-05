"""Structural, latency and clean-sanity diagnostics for Proposed RGB/Shape v1."""
from __future__ import annotations

import contextlib, json, subprocess, time
from pathlib import Path

import torch, yaml
from torch.utils.data import DataLoader
from thop import profile

from .baseline import BaselineDetector
from .coco_eval_core import evaluate_model
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape import ProposedRGBShapeV1, ProposedV1Config

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "main_80class" / "proposed_rgb_shape_v1_diagnostic"


def latency(model, x, warmup=10, repetitions=30):
    with torch.inference_mode():
        for _ in range(warmup): model(x)
        torch.cuda.synchronize(); forward=[]
        for _ in range(repetitions):
            start=time.perf_counter(); model(x); torch.cuda.synchronize(); forward.append((time.perf_counter()-start)*1000)
        for _ in range(warmup): model.predict(x, confidence=.25, iou=.7, classes=[0])
        torch.cuda.synchronize(); nms=[]
        for _ in range(repetitions):
            start=time.perf_counter(); model.predict(x, confidence=.25, iou=.7, classes=[0]); torch.cuda.synchronize(); nms.append((time.perf_counter()-start)*1000)
    def stats(values):
        t=torch.tensor(values); med=float(t.median())
        return {"median_ms":med,"mean_ms":float(t.mean()),"p95_ms":float(torch.quantile(t,.95)),"fps":1000/med}
    return {"forward":stats(forward),"forward_plus_person_nms":stats(nms)}


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    OUT.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=ProposedRGBShapeV1(ProposedV1Config(weights=str(ROOT / "yolo11n.pt"))).to(device)
    model.eval(); torch.cuda.reset_peak_memory_stats()
    x=torch.rand(1,3,640,640,device=device); output=model(x)
    diagnostic=dict(model.last_diagnostics)
    raw=output[0] if isinstance(output,tuple) else output
    diagnostic["detection_output"] = tuple(raw.shape)
    macs,_=profile(model,inputs=(x,),verbose=False)
    timings=latency(model,x)
    peak=torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0

    model.train(); batch_x=torch.rand(2,3,128,128,device=device)
    batch={"img":batch_x,"batch_idx":torch.tensor([0,0,1],device=device),"cls":torch.tensor([[0.],[0.],[0.]],device=device),"bboxes":torch.tensor([[.4,.5,.2,.5],[.7,.5,.15,.4],[.5,.5,.3,.6]],device=device)}
    loss,_=model.loss(batch,model(batch_x)); loss.sum().backward()
    gradients={
        "rgb_backbone":sum(float(p.grad.norm()) for p in model.detector.parameters() if p.grad is not None),
        "shape_backbone":sum(float(p.grad.norm()) for p in model.shape_backbone.parameters() if p.grad is not None),
        "fusion":sum(float(p.grad.norm()) for p in model.fusion.parameters() if p.grad is not None),
    }

    cfg=yaml.safe_load((ROOT/"dataset_config.main80.yaml").read_text(encoding="utf-8"))
    val=cfg["validation"]["clean"]
    images=(ROOT/Path(val["images"])).resolve(); annotations=(ROOT/Path(val["annotations"])).resolve()
    ds=Coco80DetectionDataset(CocoPersonConfig(images,annotations,image_size=640,max_images=50,patch_mode="clean",target_policy="person",required_category_ids=(1,)))
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,collate_fn=coco_person_collate)
    baseline=BaselineDetector(str(ROOT/"yolo11n.pt")).to(device).eval(); model.eval()
    with (OUT/"coco_eval.log").open("w",encoding="utf-8") as log, contextlib.redirect_stdout(log):
        baseline_metrics=evaluate_model(baseline,loader,annotations,ds,device,640,"clean",confidence=.001,iou_threshold=.7)
        proposed_metrics=evaluate_model(model,loader,annotations,ds,device,640,"clean",confidence=.001,iou_threshold=.7)
    commit=subprocess.check_output(["git","-c",f"safe.directory={ROOT}","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    result={"git_commit":commit,"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU","architecture":"ProposedRGBShapeV1","fusion_scale":"P4/stride16/layer19","fusion_type":"RGB + learnable_alpha * Conv1x1(shape)","initialization":model.initialization_report,"tensor_shapes":diagnostic,"parameter_count":{"total":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),"additional":model.initialization_report["new_parameter_count"],"gflops":2*macs/1e9},"latency":timings,"peak_vram_gib":peak,"gradient_norms":gradients,"loss":float(loss.sum().detach()),"clean_sanity_50":{"pretrained":{k:baseline_metrics[k] for k in ("person_ap","person_ap50","person_ap75","person_recall","person_gt")},"proposed_initial":{k:proposed_metrics[k] for k in ("person_ap","person_ap50","person_ap75","person_recall","person_gt")}}}
    result["seed"] = 42
    (OUT/"diagnostic.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
