"""Reusable official COCO evaluation for training-time validation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .metrics import inverse_letterbox_xyxy
from .metrics import greedy_person_match_metrics


def evaluate_model(model: Any, loader: Any, annotations: Path, dataset: Any, device: torch.device, image_size: int, attack_mode: str = "clean", confidence: float = 0.001, iou_threshold: float = 0.7) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    class_to_category = {class_index: category_id for category_id, class_index in dataset.category_id_to_class.items()}
    predictions: list[dict[str, object]] = []
    attacked_total = attacked_detected = 0
    person_gt = person_matched = person_predictions_total = 0
    person_matched75 = 0
    matched_ious: list[float] = []
    gt_ious_with_misses: list[float] = []
    person_match_status: dict[str, list[bool]] = {}
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
                item_mask = batch["batch_idx"] == index
                item_boxes = batch["bboxes"][item_mask]
                item_classes = batch["cls"][item_mask].reshape(-1)
                gt_boxes = []
                for gt_box, gt_class in zip(item_boxes, item_classes):
                    if int(gt_class.item()) != 0: continue
                    x,y,w,h = gt_box; size=image_size
                    letterbox = torch.tensor([[(x-w/2)*size,(y-h/2)*size,(x+w/2)*size,(y+h/2)*size]])
                    gt_boxes.append(inverse_letterbox_xyxy(letterbox,batch["original_size"][index],batch["letterbox_scale"][index],batch["letterbox_pad"][index])[0])
                person_gt += len(gt_boxes)
                person_mask = (detection[:,5].long().cpu()==0) & (detection[:,4].cpu()>=.25)
                person_predictions = restored[person_mask]
                person_scores = detection[:,4].cpu()[person_mask]
                match = greedy_person_match_metrics(torch.stack(gt_boxes) if gt_boxes else torch.empty((0,4)), person_predictions, person_scores)
                person_matched += int(match["recall_iou50"] * len(gt_boxes))
                person_predictions_total += int(match["prediction_count"])
                person_matched75 += int(match["recall_iou75"] * len(gt_boxes))
                matched_ious.extend(match["matched_ious"].tolist())
                gt_ious_with_misses.extend(match["gt_iou_with_misses"].tolist())
                person_match_status[str(int(batch["image_id"][index]))] = [
                    gt_index in match["matched_gt_indices_iou50"] for gt_index in range(len(gt_boxes))
                ]
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
    precision=evaluator.eval["precision"]; recall=evaluator.eval["recall"]; class_ap={}; class_ap50={}; class_ap75={}
    for index,name in enumerate(dataset.class_names):
        values=precision[:,:,index,0,-1]; valid=values[values>-1]; class_ap[name]=float(valid.mean()) if valid.size else 0.0
        v50=precision[0,:,index,0,-1]; v75=precision[5,:,index,0,-1]; v50=v50[v50>-1]; v75=v75[v75>-1]; class_ap50[name]=float(v50.mean()) if v50.size else 0.0; class_ap75[name]=float(v75.mean()) if v75.size else 0.0
    person_index=dataset.class_names.index("person") if "person" in dataset.class_names else None
    person_recall_values=recall[:,person_index,0,-1] if person_index is not None else []
    person_ar100=float(person_recall_values[person_recall_values>-1].mean()) if person_index is not None and (person_recall_values>-1).any() else 0.0
    iou_tensor = torch.tensor(matched_ious, dtype=torch.float32)
    gt_iou_tensor = torch.tensor(gt_ious_with_misses, dtype=torch.float32)
    precision_iou50 = person_matched / person_predictions_total if person_predictions_total else 0.0
    recall_iou50 = person_matched / person_gt if person_gt else 0.0
    f1_iou50 = 2 * precision_iou50 * recall_iou50 / (precision_iou50 + recall_iou50) if precision_iou50 + recall_iou50 else 0.0
    return {"attack_mode":attack_mode,"ap":float(evaluator.stats[0]),"ap50":float(evaluator.stats[1]),"ap75":float(evaluator.stats[2]),"person_ap":class_ap.get("person"),"person_ap50":class_ap50.get("person"),"person_ap75":class_ap75.get("person"),"person_ar100":person_ar100,"person_recall":recall_iou50,"person_recall_iou50_conf025":recall_iou50,"person_recall_iou75_conf025":person_matched75/person_gt if person_gt else 0.0,"precision_iou50_conf025":precision_iou50,"f1_iou50_conf025":f1_iou50,"fp_per_image_conf025":(person_predictions_total-person_matched)/len(dataset) if len(dataset) else 0.0,"mean_matched_iou":float(iou_tensor.mean()) if len(iou_tensor) else 0.0,"median_matched_iou":float(iou_tensor.median()) if len(iou_tensor) else 0.0,"std_matched_iou":float(iou_tensor.std(unbiased=False)) if len(iou_tensor) else 0.0,"matched_count":int(len(iou_tensor)),"mean_gt_iou_with_misses":float(gt_iou_tensor.mean()) if len(gt_iou_tensor) else 0.0,"person_gt":person_gt,"person_predictions_conf025":person_predictions_total,"person_match_status":person_match_status,"class_ap":class_ap,"attacked_object_detection_rate":attacked_detected/attacked_total if attacked_total else None,"failure_rate":1-attacked_detected/attacked_total if attacked_total else None,"images":len(dataset),"seconds":time.perf_counter()-started}
