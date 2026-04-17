import contextlib
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchvision.ops import box_iou


# Shared metric, plotting, and visualization helpers for the SVHN detector.


CLASS_IDS = list(range(1, 11))
DIGIT_LABELS = [str(i % 10) for i in CLASS_IDS]


def category_id_to_digit(category_id):
    return int(category_id) % 10


def _xyxy_to_xywh(boxes):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    converted = []
    for box in boxes:
        x1, y1, x2, y2 = box
        converted.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    return converted


def _ensure_prediction_maps(predictions, targets):
    # Metrics are image-aligned. Missing predictions are represented by empty
    # tensors so images with no detections still count as false negatives.
    pred_by_image = {
        pred["image_id"]: {
            "image_id": pred["image_id"],
            "boxes": pred["boxes"],
            "labels": pred["labels"],
            "scores": pred["scores"],
        }
        for pred in predictions
    }
    target_by_image = {target["image_id"]: target for target in targets}

    for image_id, target in target_by_image.items():
        if image_id not in pred_by_image:
            pred_by_image[image_id] = {
                "image_id": image_id,
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            }

    return pred_by_image, target_by_image


def calculate_metrics(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate precision, recall, F1, confusion matrix, and per-class metrics.
    """
    pred_by_image, target_by_image = _ensure_prediction_maps(predictions, targets)
    common_ids = sorted(set(pred_by_image.keys()).intersection(set(target_by_image.keys())))

    all_true_labels = []
    all_pred_labels = []

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    per_class_counts = {
        class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in CLASS_IDS
    }

    for img_id in common_ids:
        pred = pred_by_image[img_id]
        target = target_by_image[img_id]

        mask = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = pred["labels"][mask]
        pred_scores = pred["scores"][mask]

        target_boxes = target["boxes"]
        target_labels = target["labels"]

        matched_targets = set()
        matched_predictions = set()

        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            iou_matrix = box_iou(pred_boxes, target_boxes)
            if iou_matrix.numel() > 0:
                candidate_pairs = []
                for pred_idx in range(iou_matrix.shape[0]):
                    for tgt_idx in range(iou_matrix.shape[1]):
                        candidate_pairs.append(
                            (
                                iou_matrix[pred_idx, tgt_idx].item(),
                                pred_scores[pred_idx].item(),
                                pred_idx,
                                tgt_idx,
                            )
                        )

                candidate_pairs.sort(reverse=True, key=lambda item: (item[0], item[1]))

                # Greedily match the best IoU pairs first. A label mismatch still
                # consumes both boxes and becomes one FP plus one FN.
                for iou_value, _, pred_idx, tgt_idx in candidate_pairs:
                    if iou_value < iou_threshold:
                        break
                    if pred_idx in matched_predictions or tgt_idx in matched_targets:
                        continue
                    matched_predictions.add(pred_idx)
                    matched_targets.add(tgt_idx)

                    pred_label = int(pred_labels[pred_idx].item())
                    target_label = int(target_labels[tgt_idx].item())

                    all_true_labels.append(target_label)
                    all_pred_labels.append(pred_label)

                    if pred_label == target_label:
                        true_positives += 1
                        per_class_counts[target_label]["tp"] += 1
                    else:
                        false_positives += 1
                        false_negatives += 1
                        per_class_counts[pred_label]["fp"] += 1
                        per_class_counts[target_label]["fn"] += 1

        for pred_idx, pred_label in enumerate(pred_labels.tolist()):
            if pred_idx not in matched_predictions:
                false_positives += 1
                per_class_counts[int(pred_label)]["fp"] += 1

        for tgt_idx, target_label in enumerate(target_labels.tolist()):
            if tgt_idx not in matched_targets:
                false_negatives += 1
                per_class_counts[int(target_label)]["fn"] += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    cm = (
        confusion_matrix(all_true_labels, all_pred_labels, labels=CLASS_IDS)
        if all_true_labels and all_pred_labels
        else np.zeros((10, 10))
    )

    per_class_metrics = {}
    for class_id in CLASS_IDS:
        tp = per_class_counts[class_id]["tp"]
        fp = per_class_counts[class_id]["fp"]
        fn = per_class_counts[class_id]["fn"]
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall) > 0
            else 0.0
        )
        per_class_metrics[class_id] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
    }


def compute_map_metrics(predictions, targets):
    """
    Compute COCO-style mAP, AP50, and AP75 from prediction/target dictionaries.
    """
    pred_by_image, target_by_image = _ensure_prediction_maps(predictions, targets)
    image_ids = sorted(target_by_image.keys())

    # Build an in-memory COCO object so pycocotools can compute standard AP
    # values without writing temporary annotation files.
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": int(image_id)} for image_id in image_ids],
        "annotations": [],
        "categories": [{"id": class_id, "name": str(class_id - 1)} for class_id in CLASS_IDS],
    }

    annotation_id = 1
    for image_id in image_ids:
        target = target_by_image[image_id]
        boxes_xywh = _xyxy_to_xywh(target["boxes"])
        labels = target["labels"].tolist()
        for box, label in zip(boxes_xywh, labels):
            coco_gt.dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": box,
                    "area": float(box[2] * box[3]),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco_gt.createIndex()

    results = []
    for image_id, pred in pred_by_image.items():
        boxes_xywh = _xyxy_to_xywh(pred["boxes"])
        labels = pred["labels"].tolist()
        scores = pred["scores"].tolist()
        for box, label, score in zip(boxes_xywh, labels, scores):
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": box,
                    "score": float(score),
                }
            )

    if len(results) == 0:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return {
        "mAP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }


def plot_losses(train_losses, val_losses, val_metrics, save_path):
    """
    Plot train/val losses, validation metrics, and mAP curves.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(
        epochs,
        [loss["loss"] for loss in train_losses],
        "b-o",
        label="Train Total Loss",
    )
    axes[0].plot(
        epochs,
        [loss["loss"] for loss in val_losses],
        "k--o",
        label="Val Total Loss",
    )

    preferred_loss_keys = ["loss_ce", "loss_bbox", "loss_giou", "cardinality_error"]
    colors = ["g-", "r-", "m-", "c-"]
    for key, color in zip(preferred_loss_keys, colors):
        if key in train_losses[0]:
            axes[0].plot(
                epochs,
                [loss.get(key, 0.0) for loss in train_losses],
                color[:-1] + "-o",
                label=f"Train {key}",
            )
            axes[0].plot(
                epochs,
                [loss.get(key, 0.0) for loss in val_losses],
                color[:-1] + "--o",
                label=f"Val {key}",
            )

    axes[0].set_title("Train vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(True)

    axes[1].plot(epochs, [m["precision"] for m in val_metrics], "b-o", label="Precision")
    axes[1].plot(epochs, [m["recall"] for m in val_metrics], "g-o", label="Recall")
    axes[1].plot(epochs, [m["f1"] for m in val_metrics], "r-o", label="F1 Score")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, [m["mAP"] for m in val_metrics], "m-o", label="mAP")
    axes[2].plot(epochs, [m["AP50"] for m in val_metrics], "c-o", label="AP50")
    axes[2].plot(epochs, [m["AP75"] for m in val_metrics], "y-o", label="AP75")
    axes[2].set_title("Validation mAP")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AP")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(epochs, [m["true_positives"] for m in val_metrics], "g-o", label="TP")
    axes[3].plot(epochs, [m["false_positives"] for m in val_metrics], "r-o", label="FP")
    axes[3].plot(epochs, [m["false_negatives"] for m in val_metrics], "b-o", label="FN")
    axes[3].set_title("Validation TP / FP / FN")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Count")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DIGIT_LABELS)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_metrics(per_class_metrics, save_path):
    class_ids = CLASS_IDS
    x = np.arange(len(class_ids))
    width = 0.25

    precisions = [per_class_metrics[class_id]["precision"] for class_id in class_ids]
    recalls = [per_class_metrics[class_id]["recall"] for class_id in class_ids]
    f1_scores = [per_class_metrics[class_id]["f1"] for class_id in class_ids]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width=width, label="Precision")
    ax.bar(x, recalls, width=width, label="Recall")
    ax.bar(x + width, f1_scores, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(DIGIT_LABELS)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class Metrics")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_prediction_diagnostics(predictions, targets, save_path, score_threshold=0.5):
    pred_by_image, target_by_image = _ensure_prediction_maps(predictions, targets)
    common_ids = sorted(set(pred_by_image.keys()).intersection(set(target_by_image.keys())))

    score_values = []
    pred_counts = []
    gt_counts = []

    # Compare confidence and count distributions to quickly spot over-detection
    # or score-threshold problems.
    for image_id in common_ids:
        pred = pred_by_image[image_id]
        target = target_by_image[image_id]
        keep = pred["scores"] >= score_threshold
        score_values.extend(pred["scores"][keep].tolist())
        pred_counts.append(int(keep.sum().item()))
        gt_counts.append(int(len(target["boxes"])))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].hist(score_values, bins=20, color="steelblue", edgecolor="black")
    axes[0].set_title("Prediction Score Distribution")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1)
    if not score_values:
        axes[0].text(
            0.5,
            0.5,
            f"No predictions above threshold ({score_threshold:.2f})",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    axes[0].grid(True)

    max_count = max(pred_counts + gt_counts + [1])
    bins = np.arange(0, max_count + 2) - 0.5
    axes[1].hist(pred_counts, bins=bins, alpha=0.7, label="Predicted boxes / image")
    axes[1].hist(gt_counts, bins=bins, alpha=0.7, label="Ground-truth boxes / image")
    axes[1].set_title("Box Count Distribution per Image")
    axes[1].set_xlabel("Number of Boxes")
    axes[1].set_ylabel("Images")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def format_per_class_metrics(per_class_metrics):
    lines = []
    for class_id in CLASS_IDS:
        metrics = per_class_metrics[class_id]
        lines.append(
            "Digit {} -> P: {:.4f}, R: {:.4f}, F1: {:.4f}, TP: {}, FP: {}, FN: {}".format(
                class_id - 1,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["tp"],
                metrics["fp"],
                metrics["fn"],
            )
        )
    return lines


def create_improved_task2_predictions(predictions):
    # This older helper applies a stricter fixed score threshold before reading
    # digits left to right. The main inference path now uses create_task2_predictions.
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        predictions_by_image[pred["image_id"]].append(pred)

    results = []
    for img_id, preds in predictions_by_image.items():
        preds = [p for p in preds if p["score"] > 0.6]

        if len(preds) == 0:
            results.append({"image_id": img_id, "pred_label": -1})
            continue

        def sort_key(pred):
            x, y, _, _ = pred["bbox"]
            return x + 0.01 * y

        preds.sort(key=sort_key)
        digits = [str(category_id_to_digit(p["category_id"])) for p in preds]
        num = int("".join(digits))
        results.append({"image_id": img_id, "pred_label": num})

    return results


def visualize_predictions(
    model, dataset, device, indices=None, num_samples=5, save_dir=None, score_threshold=0.5
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if indices is None:
        indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        image_id = target["image_id"].item()
        image_tensor = image.to(device)

        with torch.no_grad():
            predictions = model([image_tensor])
            prediction = predictions[0]

        # Undo ImageNet normalization before drawing boxes on the saved image.
        inv_normalize = torchvision.transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image_unnorm = inv_normalize(image.detach().cpu()).clamp(0.0, 1.0)
        image_pil = torchvision.transforms.functional.to_pil_image(image_unnorm)
        draw = ImageDraw.Draw(image_pil)

        for box, label in zip(target["boxes"], target["labels"]):
            box = box.numpy()
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="green", width=2)
            draw.text(
                (box[0], box[1] - 10),
                f"{category_id_to_digit(label.item())}",
                fill="green",
            )

        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if score > score_threshold:
                box = box.cpu().numpy()
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
                draw.text(
                    (box[0], box[1] - 10),
                    f"{category_id_to_digit(label.item())} ({score:.2f})",
                    fill="red",
                )

        image_pil.save(os.path.join(save_dir, f"pred_img{image_id}.png"))

    print(f"Saved {len(indices)} visualization images to {save_dir}")
