#!/usr/bin/env python3
"""
ensemble.py - Ensemble inference for Cell Instance Segmentation
===============================================================

This script runs late-fusion inference for two independently trained
Mask R-CNN checkpoints, such as ResNet101-FPN and ConvNeXtV2-FPN.

Typical usage:
    python ensemble.py \
        --data_root data \
        --ckpt_a outputs_resnet101/best.pth \
        --ckpt_b outputs_convnextv2/best.pth \
        --out_file test-results-ensemble.json

Validation evaluation:
    python ensemble.py \
        --mode val \
        --data_root data \
        --ckpt_a outputs_resnet101/best.pth \
        --ckpt_b outputs_convnextv2/best.pth \
        --eval_out outputs_ensemble/ensemble_eval.json
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import boxes as box_ops

from datasets import EnhancedCellDataset, collate_fn
from inference import (
    get_checkpoint_state,
    infer_backbone_from_state_dict,
    process_large_image_with_tiling,
    process_standard_image,
)
from models import CascadeMaskRCNN
from utils import coco_bbox_xyxy_to_xywh, decode_maskobj, encode_mask


CLASSES = ["background", "class1", "class2", "class3", "class4"]
NUM_CLASSES = len(CLASSES)


def adapt_legacy_checkpoint_state(state_dict):
    """
    Convert older cascade-head checkpoint keys to the current single-head model.

    Older experiments saved top-level keys such as ``box_predictors.0.*``.
    The current model uses ``box_predictor.*`` and shares it through
    ``roi_heads.box_predictor``. For inference, using the first cascade stage is
    the safest compatibility mapping because it matches the original predictor
    dimensions and avoids silently dropping all classifier/regressor weights.
    """
    adapted = dict(state_dict)

    if "box_predictor.cls_score.weight" not in adapted:
        legacy_prefix = "box_predictors.0."
        legacy_keys = [key for key in adapted if key.startswith(legacy_prefix)]
        if legacy_keys:
            print("Detected legacy cascade box heads; mapping box_predictors.0 -> box_predictor.")
            for key in legacy_keys:
                new_key = key.replace(legacy_prefix, "box_predictor.", 1)
                adapted[new_key] = adapted[key]
    for key in list(adapted.keys()):
        if key.startswith("box_predictors."):
            del adapted[key]

    if "mask_predictor.conv5_mask.weight" not in adapted:
        legacy_prefix = "mask_predictors.0."
        legacy_keys = [key for key in adapted if key.startswith(legacy_prefix)]
        if legacy_keys:
            print("Detected legacy cascade mask heads; mapping mask_predictors.0 -> mask_predictor.")
            for key in legacy_keys:
                new_key = key.replace(legacy_prefix, "mask_predictor.", 1)
                adapted[new_key] = adapted[key]
    for key in list(adapted.keys()):
        if key.startswith("mask_predictors."):
            del adapted[key]

    return adapted


def load_model(ckpt_path, device, fallback_backbone):
    """Load a checkpoint and rebuild its matching backbone."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = adapt_legacy_checkpoint_state(get_checkpoint_state(checkpoint))
    backbone = None
    if isinstance(checkpoint, dict):
        backbone = checkpoint.get("backbone")
    if backbone is None:
        backbone = infer_backbone_from_state_dict(state_dict)
    if backbone is None:
        backbone = fallback_backbone

    model = CascadeMaskRCNN(NUM_CLASSES, backbone=backbone).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {ckpt_path} ({backbone})")
    return model, backbone


def record_to_box_xyxy(record):
    """Convert a COCO xywh bbox record to xyxy."""
    x, y, w, h = record["bbox"]
    return [x, y, x + w, y + h]


def mask_iou(record_a, record_b):
    """Compute IoU between two encoded instance masks."""
    mask_a = decode_maskobj(record_a["segmentation"]).astype(bool)
    mask_b = decode_maskobj(record_b["segmentation"]).astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def fuse_records(records, box_nms_thresh=0.5, mask_iou_thresh=0.6, max_dets=2000):
    """
    Fuse predictions from multiple models with class-wise box NMS plus mask IoU.

    For highly overlapping same-class instances, the higher-scoring record is kept.
    The score is mildly boosted when another model supports the same object.
    """
    if not records:
        return []

    kept_records = []
    for cls_id in range(1, NUM_CLASSES):
        cls_records = [r.copy() for r in records if int(r["category_id"]) == cls_id]
        if not cls_records:
            continue

        # First remove obvious duplicate boxes, then verify mask overlap.
        boxes = torch.tensor([record_to_box_xyxy(r) for r in cls_records], dtype=torch.float32)
        scores = torch.tensor([float(r["score"]) for r in cls_records], dtype=torch.float32)
        keep = box_ops.nms(boxes, scores, box_nms_thresh).tolist()

        selected = []
        for idx in keep:
            candidate = cls_records[idx]
            duplicate = False
            for kept in selected:
                if mask_iou(candidate, kept) >= mask_iou_thresh:
                    kept["score"] = min(1.0, max(float(kept["score"]), float(candidate["score"])) + 0.03)
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)

        kept_records.extend(selected)

    kept_records.sort(key=lambda r: float(r["score"]), reverse=True)
    return kept_records[:max_dets]


def infer_image_ensemble(models, path, img_id, args, device):
    """Run all models on one test image and fuse their COCO-format records."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[..., :3]

    all_records = []
    with torch.no_grad():
        for model in models:
            if max(img.shape[:2]) > args.max_dim:
                model_records = process_large_image_with_tiling(model, img, img_id, args, device)
            else:
                model_records = process_standard_image(model, img, img_id, args, device)
            all_records.extend(model_records)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return fuse_records(
        all_records,
        box_nms_thresh=args.ensemble_nms_thresh,
        mask_iou_thresh=args.mask_iou_thresh,
        max_dets=args.max_dets,
    )


def run_test_inference(models, args, device):
    """Generate fused test predictions in CodaBench/COCO result format."""
    data_root = Path(args.data_root)
    test_dir = data_root / "test"
    if not test_dir.exists():
        test_dir = data_root / "test_release"
    test_imgs = sorted(test_dir.glob("*.tif"))

    with open(data_root / "test_image_name_to_ids.json", encoding="utf-8") as f:
        imgname2id_data = json.load(f)
    if isinstance(imgname2id_data, list):
        imgname2id = {item["file_name"]: item["id"] for item in imgname2id_data}
    else:
        imgname2id = imgname2id_data

    records = []
    for idx, path in enumerate(test_imgs, 1):
        img_id = int(imgname2id[path.name])
        print(f"[{idx}/{len(test_imgs)}] Ensemble inference: {path.name}")
        records.extend(infer_image_ensemble(models, path, img_id, args, device))

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Saved ensemble predictions to {out_path} ({len(records)} instances)")


def predict_tensor_tta(model, image, args, device):
    """Run horizontal-flip TTA on one validation tensor and return raw predictions."""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        pred_orig = model([image])[0]

        flipped = torch.flip(image, [2])
        pred_flip = model([flipped])[0]
        width = image.shape[2]

        boxes_flip = pred_flip["boxes"].clone()
        boxes_flip[:, [0, 2]] = width - boxes_flip[:, [2, 0]]
        masks_flip = torch.flip(pred_flip["masks"], [3])

        pred = {
            "boxes": torch.cat([pred_orig["boxes"], boxes_flip]).cpu(),
            "scores": torch.cat([pred_orig["scores"], pred_flip["scores"]]).cpu(),
            "labels": torch.cat([pred_orig["labels"], pred_flip["labels"]]).cpu(),
            "masks": torch.cat([pred_orig["masks"], masks_flip]).cpu(),
        }
    return pred


def raw_prediction_to_records(pred, img_id, score_thresh, nms_thresh):
    """Convert raw model output to COCO result records after class-wise NMS."""
    records = []
    for cls_id in range(1, NUM_CLASSES):
        cls_indices = (pred["labels"] == cls_id).nonzero(as_tuple=True)[0]
        if len(cls_indices) == 0:
            continue
        keep = box_ops.nms(
            pred["boxes"][cls_indices],
            pred["scores"][cls_indices],
            nms_thresh,
        )
        for idx in cls_indices[keep]:
            score = float(pred["scores"][idx].item())
            if score < score_thresh:
                continue
            mask = (pred["masks"][idx, 0] > 0.5).numpy()
            if not np.any(mask):
                continue
            records.append(
                {
                    "image_id": img_id,
                    "category_id": int(pred["labels"][idx].item()),
                    "score": score,
                    "bbox": coco_bbox_xyxy_to_xywh(pred["boxes"][idx]),
                    "segmentation": encode_mask(mask),
                }
            )
    return records


def target_to_gt_records(target):
    """Convert one validation target to COCO ground-truth annotations."""
    img_id = int(target["image_id"].item())
    records = []
    for mask, box, label in zip(target["masks"], target["boxes"], target["labels"]):
        records.append(
            {
                "id": 0,
                "image_id": img_id,
                "category_id": int(label.item()),
                "segmentation": encode_mask(mask.numpy()),
                "area": float(mask.sum().item()),
                "bbox": coco_bbox_xyxy_to_xywh(box),
                "iscrowd": 0,
            }
        )
    return records


def evaluate_records(pred_records, gt_records, image_info, out_json=None):
    """Run COCO segmentation evaluation and optionally save summary JSON."""
    for ann_id, record in enumerate(gt_records, 1):
        record["id"] = ann_id

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": image_info,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES[1:], 1)],
        "annotations": gt_records,
    }
    coco_gt.createIndex()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp.name, "w", encoding="utf-8") as f:
        json.dump(pred_records, f)

    metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "predictions": len(pred_records)}
    try:
        coco_dt = coco_gt.loadRes(tmp.name)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics.update(
            {
                "AP": float(coco_eval.stats[0]),
                "AP50": float(coco_eval.stats[1]),
                "AP75": float(coco_eval.stats[2]),
            }
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}")

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved validation metrics to {out_path}")

    print(
        "Validation ensemble: "
        f"AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}, AP75={metrics['AP75']:.4f}"
    )
    return metrics


def tensor_to_uint8_image(image):
    """Convert a CHW float tensor image to uint8 RGB."""
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def draw_instance_contours(image, records, color, thickness=2):
    """Draw instance mask contours on an RGB image copy."""
    canvas = image.copy()
    for record in records:
        mask = decode_maskobj(record["segmentation"]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, thickness)
    return canvas


def add_panel_title(image, title, color):
    """Add a readable title to a visualization panel."""
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    return canvas


def save_model_comparison_visualization(
    image,
    gt_records,
    model_records,
    ensemble_records,
    out_dir,
    image_id,
):
    """Save GT / model A / model B / ensemble comparison for qualitative analysis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = tensor_to_uint8_image(image)
    panels = [
        add_panel_title(
            draw_instance_contours(rgb, gt_records, (40, 220, 80), thickness=2),
            "Ground Truth",
            (40, 220, 80),
        )
    ]

    panel_specs = [
        ("ResNet101", model_records[0], (60, 170, 255)),
        ("ConvNeXtV2", model_records[1], (255, 190, 60)),
        ("Ensemble", ensemble_records, (255, 80, 40)),
    ]
    for title, records, color in panel_specs:
        panels.append(add_panel_title(draw_instance_contours(rgb, records, color, 2), title, color))

    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    comparison = np.concatenate([top, bottom], axis=0)

    out_path = out_dir / f"qual_comparison_{image_id}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def run_validation(models, args, device):
    """Evaluate the ensemble on the validation split."""
    data_root = Path(args.data_root)
    with open(data_root / "split_info.json", encoding="utf-8") as f:
        split = json.load(f)

    dataset = EnhancedCellDataset(data_root, split["val"], args.max_dim, aug=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    pred_records = []
    gt_records = []
    image_info = []
    for idx, (images, targets) in enumerate(loader, 1):
        image = images[0]
        target = targets[0]
        img_id = int(target["image_id"].item())
        image_info.append({"id": img_id, "height": image.shape[1], "width": image.shape[2]})
        image_gt_records = target_to_gt_records(target)
        gt_records.extend(image_gt_records)

        all_records = []
        per_model_records = []
        for model in models:
            pred = predict_tensor_tta(model, image, args, device)
            model_records = raw_prediction_to_records(
                pred,
                img_id,
                score_thresh=args.val_score_thresh,
                nms_thresh=args.nms_thresh,
            )
            per_model_records.append(model_records)
            all_records.extend(model_records)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        image_pred_records = fuse_records(
            all_records,
            box_nms_thresh=args.ensemble_nms_thresh,
            mask_iou_thresh=args.mask_iou_thresh,
            max_dets=args.max_dets,
        )
        pred_records.extend(image_pred_records)

        if args.vis_dir and idx <= args.num_vis:
            if len(per_model_records) >= 2:
                # These panels are used as qualitative evidence in the report.
                save_model_comparison_visualization(
                    image,
                    image_gt_records,
                    per_model_records,
                    image_pred_records,
                    args.vis_dir,
                    image_id=img_id,
                )

        if idx % 5 == 0 or idx == len(loader):
            print(f"Validated {idx}/{len(loader)} images")

    evaluate_records(pred_records, gt_records, image_info, args.eval_out)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Late-fusion ensemble inference")
    parser.add_argument("--mode", choices=["test", "val"], default="test")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--ckpt_b", type=str, required=True)
    parser.add_argument("--backbone_a", type=str, default="resnet101")
    parser.add_argument("--backbone_b", type=str, default="convnextv2_base")
    parser.add_argument("--out_file", type=str, default="test-results-ensemble.json")
    parser.add_argument("--eval_out", type=str, default="outputs_ensemble/ensemble_eval.json")
    parser.add_argument("--max_dim", type=int, default=1024)
    parser.add_argument("--max_dets", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--score_thresh", type=float, default=0.1)
    parser.add_argument("--val_score_thresh", type=float, default=0.05)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--ensemble_nms_thresh", type=float, default=0.5)
    parser.add_argument("--mask_iou_thresh", type=float, default=0.6)
    parser.add_argument("--vis_dir", type=str, default=None, help="Optional validation overlay output dir")
    parser.add_argument("--num_vis", type=int, default=8, help="Number of validation overlays to save")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    """Entry point."""
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model_a, _ = load_model(args.ckpt_a, device, args.backbone_a)
    model_b, _ = load_model(args.ckpt_b, device, args.backbone_b)
    models = [model_a, model_b]

    if args.mode == "test":
        run_test_inference(models, args, device)
    else:
        run_validation(models, args, device)


if __name__ == "__main__":
    main()
