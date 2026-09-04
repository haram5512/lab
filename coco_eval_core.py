"""Reusable official COCO evaluation for training-time validation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .metrics import inverse_letterbox_xyxy


def evaluate_model(model: Any, loader: Any, annotations: Path, dataset: Any, device: torch.device, image_size: int, attack_mode: str = "clean", confidence: float = 0.001, iou_threshold: float = 0.7) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    class_to_category = {class_index: category_id for category_id, class_index in dataset.category_id_to_class.items()}
    predictions: list[dict[str, object]] = []
    attacked_total = attacked_detected = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=device.type == "cuda")
            if hasattr(model, "predict"):
                detections = model.predict(images, confidence=confidence, iou=iou_threshold, classes=None)
            else:
                detections = model.predict_tensor(images, confidence=confidence, iou=iou_threshold)
            for index, detection in enumerate(detections):
                restored = inverse_letterbox_xyxy(detection[:, :4].cpu(), batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index])
                for box, score, class_index in zip(restored, detection[:, 4].cpu(), detection[:, 5].long().cpu()):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    predictions.append({"image_id": int(batch["image_id"][index]), "category_id": int(class_to_category[int(class_index)]), "bbox": [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)], "score": float(score)})
                if attack_mode != "clean" and batch["patch_target"][index] is not None:
                    target = torch.tensor(batch["patch_target"][index], dtype=torch.float32)
                    target_xyxy = torch.tensor([(target[0]-target[2]/2)*image_size, (target[1]-target[3]/2)*image_size, (target[0]+target[2]/2)*image_size, (target[1]+target[3]/2)*image_size]).reshape(1,4)
                    target_xyxy = inverse_letterbox_xyxy(target_xyxy, batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index])[0]
                    attacked_total += 1
                    candidates = detection[detection[:, 5].long().cpu() == int(batch["patch_target_class"][index])]
                    if len(candidates):
                        candidate_boxes = inverse_letterbox_xyxy(candidates[:, :4].cpu(), batch["original_size"][index], batch["letterbox_scale"][index], batch["letterbox_pad"][index])
                        lt=torch.maximum(candidate_boxes[:,:2],target_xyxy[:2]); rb=torch.minimum(candidate_boxes[:,2:],target_xyxy[2:]); wh=(rb-lt).clamp(min=0); inter=wh[:,0]*wh[:,1]
                        area_t=(target_xyxy[2]-target_xyxy[0])*(target_xyxy[3]-target_xyxy[1]); area_c=(candidate_boxes[:,2]-candidate_boxes[:,0])*(candidate_boxes[:,3]-candidate_boxes[:,1])
                        if bool((inter/(area_t+area_c-inter).clamp(min=1e-9) >= .5).any()): attacked_detected += 1
    coco=COCO(str(annotations)); result=coco.loadRes(predictions) if predictions else coco.loadRes([]); evaluator=COCOeval(coco,result,"bbox"); evaluator.params.imgIds=[int(record[0]["id"]) for record in dataset.records]; evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    precision=evaluator.eval["precision"]; class_ap={}
    for index,name in enumerate(dataset.class_names):
        values=precision[:,:,index,0,-1]; valid=values[values>-1]; class_ap[name]=float(valid.mean()) if valid.size else 0.0
    return {"attack_mode":attack_mode,"ap":float(evaluator.stats[0]),"ap50":float(evaluator.stats[1]),"ap75":float(evaluator.stats[2]),"person_ap":class_ap.get("person"),"class_ap":class_ap,"attacked_object_detection_rate":attacked_detected/attacked_total if attacked_total else None,"images":len(dataset),"seconds":time.perf_counter()-started}
