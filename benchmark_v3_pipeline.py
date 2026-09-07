"""Short v3 throughput benchmark for batch/loader choices before diagnostic training."""
from __future__ import annotations
import json, time
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
from .dataset import Coco80DetectionDataset, CocoPersonConfig, coco_person_collate
from .models.proposed_rgb_shape_v3 import ProposedRGBShapeV3, ProposedV3Config

def main() -> None:
    root=Path(__file__).resolve().parent; out=root.parent/"runs"/"main_80class"/"proposed_rgb_shape_v3_yolo_backbone"; out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((root/"dataset_config.main80.yaml").read_text(encoding="utf-8")); val=cfg["validation"]["clean"]; images=(root/Path(val["images"])).resolve(); ann=(root/Path(val["annotations"])).resolve(); payload=json.loads(ann.read_text(encoding="utf-8")); cats=tuple(sorted(int(x["id"]) for x in payload["categories"]))
    common=dict(images_dir=images,annotations_file=ann,image_size=640,max_images=500,target_policy="person",required_category_ids=(),category_ids=cats,seed=7,position_jitter=.015,placement_mode="torso",torso_relative_y=.38,patch_scale_min=.95,patch_scale_max=1.05,patch_rotation_degrees=3.,patch_brightness_jitter=.03,patch_perspective_jitter=.005)
    ds=Coco80DetectionDataset(CocoPersonConfig(**common,patch_mode="clean",patch_probability=0.,patch_dirs=())); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); results=[]
    for batch_size in (8,16):
        for workers in (0,2):
            model=ProposedRGBShapeV3(ProposedV3Config(weights=str(root/"yolo11n.pt"))).to(device).train(); loader=DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=coco_person_collate,num_workers=workers,pin_memory=device.type=="cuda",persistent_workers=workers>0); iterator=iter(loader); times=[]
            for index in range(20):
                try: batch=next(iterator)
                except StopIteration: iterator=iter(loader); batch=next(iterator)
                images_t=batch["img"].to(device,non_blocking=True); started=time.perf_counter(); model.zero_grad(set_to_none=True); output=model(images_t); leaves=[]
                def collect(x):
                    if isinstance(x,torch.Tensor): leaves.append(x)
                    elif isinstance(x,dict):
                        for y in x.values(): collect(y)
                    elif isinstance(x,(list,tuple)):
                        for y in x: collect(y)
                collect(output); sum(x.float().mean() for x in leaves).backward();
                if device.type=="cuda": torch.cuda.synchronize()
                times.append(time.perf_counter()-started)
            elapsed=sum(times[2:]); results.append({"batch_size":batch_size,"num_workers":workers,"pin_memory":device.type=="cuda","mean_sec_per_batch":elapsed/max(1,len(times)-2),"images_per_sec":batch_size*(len(times)-2)/elapsed,"peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0.0}); del model
            if device.type=="cuda": torch.cuda.empty_cache()
    (out/"optimization_benchmark.json").write_text(json.dumps(results,indent=2),encoding="utf-8"); print(json.dumps(results,indent=2))
if __name__=="__main__": main()
