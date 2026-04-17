import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from inference import (
    _collect_predictions_for_image,
    _direct_predictions,
    _image_id_from_filename,
    _prepare_tta_views,
    _resolve_device,
    create_task2_predictions,
)
from utils import (
    calculate_metrics,
    compute_map_metrics,
    format_per_class_metrics,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_prediction_diagnostics,
)


# Generate report figures and summaries from training and validation logs.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final report artifacts from train.log and one or more validation logs."
    )
    parser.add_argument(
        "--train-log",
        type=str,
        default=None,
        help="Optional training log (.log / .log.json). Used for training_curves.png.",
    )
    parser.add_argument(
        "--validation-plain",
        type=str,
        nargs="+",
        default=None,
        help="Validation log(s) for the plain baseline without TTA.",
    )
    parser.add_argument(
        "--validation-flip",
        type=str,
        nargs="+",
        default=None,
        help="Validation log(s) for horizontal-flip TTA experiments.",
    )
    parser.add_argument(
        "--validation-scale",
        type=str,
        nargs="+",
        default=None,
        help="Validation log(s) for resize-scale TTA experiments.",
    )
    parser.add_argument(
        "--validation-both",
        type=str,
        nargs="+",
        default=None,
        help="Validation log(s) for combined scale + flip TTA experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where report plots and summaries are saved.",
    )
    parser.add_argument(
        "--top-case-studies",
        type=int,
        default=4,
        help="How many error cases to include in case_study.png.",
    )
    return parser.parse_args()


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def _load_val_annotations(val_ann_path: Path):
    # Keep annotations grouped by image id because later metrics and drawings are
    # computed image-by-image.
    with val_ann_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    image_info = {int(img["id"]): img["file_name"] for img in payload.get("images", [])}
    anns_by_img = defaultdict(list)
    for ann in payload.get("annotations", []):
        anns_by_img[int(ann["image_id"])].append(ann)
    return image_info, anns_by_img


def _build_targets(image_ids, anns_by_img):
    targets = []
    for image_id in image_ids:
        anns = anns_by_img.get(image_id, [])
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(_xywh_to_xyxy(ann["bbox"]))
            labels.append(int(ann["category_id"]))
        targets.append(
            {
                "image_id": int(image_id),
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            }
        )
    return targets


def _build_task2_ground_truth(image_ids, anns_by_img):
    # The ground-truth number is formed by reading digit boxes from left to right.
    rows = []
    for image_id in image_ids:
        anns = sorted(anns_by_img.get(image_id, []), key=lambda ann: ann["bbox"][0])
        if not anns:
            rows.append({"image_id": int(image_id), "gt_label": -1})
            continue
        digits = [str(int(ann["category_id"]) % 10) for ann in anns]
        rows.append({"image_id": int(image_id), "gt_label": int("".join(digits))})
    return rows


def _prediction_dict_for_metrics(predictions, image_ids):
    # Convert COCO-style prediction rows back into tensor dictionaries used by
    # the shared metric helpers.
    preds_by_image = defaultdict(list)
    for pred in predictions:
        preds_by_image[int(pred["image_id"])].append(pred)

    metric_predictions = []
    for image_id in image_ids:
        preds = preds_by_image.get(int(image_id), [])
        boxes = []
        labels = []
        scores = []
        for pred in preds:
            x, y, w, h = pred["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(pred["category_id"]))
            scores.append(float(pred["score"]))
        metric_predictions.append(
            {
                "image_id": int(image_id),
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
                "scores": torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros((0,), dtype=torch.float32),
            }
        )
    return metric_predictions


def _render_validation_predictions(val_dir, image_info, anns_by_img, preds_by_img, save_dir, score_thr, max_samples):
    save_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = sorted(image_info.keys())[:max_samples]

    for image_id in selected_ids:
        image_path = val_dir / image_info[image_id]
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Green boxes are ground truth; red boxes are model predictions.
        for ann in anns_by_img.get(image_id, []):
            x1, y1, x2, y2 = _xywh_to_xyxy(ann["bbox"])
            label = int(ann["category_id"]) % 10
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
            draw.text((x1, max(0, y1 - 10)), f"{label}", fill="green")

        for pred in preds_by_img.get(image_id, []):
            if float(pred["score"]) < score_thr:
                continue
            x, y, w, h = pred["bbox"]
            x2 = x + w
            y2 = y + h
            label = int(pred["category_id"]) % 10
            draw.rectangle([(x, y), (x2, y2)], outline="red", width=2)
            draw.text((x, max(0, y - 10)), f"{label} ({pred['score']:.2f})", fill="red")

        image.save(save_dir / f"val_pred_{image_id}.png")


def _parse_text_log(log_path: Path):
    # MMEngine text logs are not structured JSON, so regex extraction keeps the
    # training-curve code independent of a specific logger backend.
    train_pattern = re.compile(r"Epoch\(train\)\s+\[(\d+)\]\[\s*\d+/\d+\].*?\sloss:\s*([0-9.]+)")
    val_pattern = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[\d+/\d+\].*?coco/bbox_mAP:\s*([0-9.]+)\s+"
        r"coco/bbox_mAP_50:\s*([0-9.]+)\s+coco/bbox_mAP_75:\s*([0-9.]+)"
    )

    train_loss_by_epoch = defaultdict(list)
    map_by_epoch = {}
    ap50_by_epoch = {}
    ap75_by_epoch = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                train_loss_by_epoch[epoch].append(float(train_match.group(2)))
                continue

            val_match = val_pattern.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                map_by_epoch[epoch] = float(val_match.group(2))
                ap50_by_epoch[epoch] = float(val_match.group(3))
                ap75_by_epoch[epoch] = float(val_match.group(4))

    epochs = sorted(set(train_loss_by_epoch) | set(map_by_epoch) | set(ap50_by_epoch) | set(ap75_by_epoch))
    return {
        "epochs": epochs,
        "train_loss": [float(np.mean(train_loss_by_epoch[e])) if e in train_loss_by_epoch else np.nan for e in epochs],
        "mAP": [map_by_epoch.get(e, np.nan) for e in epochs],
        "AP50": [ap50_by_epoch.get(e, np.nan) for e in epochs],
        "AP75": [ap75_by_epoch.get(e, np.nan) for e in epochs],
    }


def _plot_curves_from_text_log(log_path: Path, save_path: Path):
    parsed = _parse_text_log(log_path)
    epochs = parsed["epochs"]

    if not epochs:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No epoch metrics found in text log", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].plot(epochs, parsed["train_loss"], "b-o", label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, parsed["mAP"], "m-o", label="mAP")
    axes[1].plot(epochs, parsed["AP50"], "c-o", label="AP50")
    axes[1].plot(epochs, parsed["AP75"], "y-o", label="AP75")
    axes[1].set_title("Validation AP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AP")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _run_validation_predictions(args, image_info):
    import mmcv
    from mmdet.apis import inference_detector, init_detector

    device = _resolve_device(args.device)
    checkpoint_paths = [str(Path(path).resolve()) for path in args.checkpoint]
    models = [init_detector(args.config, checkpoint_path, device=device) for checkpoint_path in checkpoint_paths]

    predictions = []
    preds_by_img = defaultdict(list)
    val_dir = (Path(args.data_dir).resolve() / "valid")
    use_single_view = len(models) == 1 and not args.tta_horizontal_flip and args.tta_scales == [1.0]

    # This path is kept for standalone report generation when validation logs
    # have not already been produced by validation.py.
    for image_id in tqdm(sorted(image_info.keys()), desc="Running validation inference"):
        image_path = val_dir / image_info[image_id]
        if not image_path.exists():
            continue

        if use_single_view:
            result = inference_detector(models[0], str(image_path))
            merged_predictions = _direct_predictions(result, score_thr=args.score_thr)
        else:
            image = mmcv.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            tta_views = _prepare_tta_views(image, args.tta_scales, args.tta_horizontal_flip)
            merged_predictions = _collect_predictions_for_image(
                models=models,
                tta_views=tta_views,
                iou_thr=args.ensemble_iou_thr,
                pre_merge_score_thr=args.pre_merge_score_thr,
            )

        for bbox, label, score in merged_predictions:
            x1, y1, x2, y2 = bbox
            pred = {
                "image_id": int(image_id),
                "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                "score": float(score),
                "category_id": int(label) + 1,
            }
            preds_by_img[int(image_id)].append(pred)
            if float(score) >= args.score_thr:
                predictions.append(pred)

    predictions.sort(key=lambda item: (item["image_id"], item["bbox"][0], -item["score"]))
    return predictions, preds_by_img


def generate_validation_report(args) -> dict:
    # Legacy entry point: run validation, compute metrics, and emit figures in
    # one command. The newer flow prefers validation.py + generate_report().
    try:
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise ImportError(
            "MMDetection is not installed. Install the Co-DETR stack first before running report.py."
        ) from exc

    register_all_modules(init_default_scope=True)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_ann_path = data_dir / "codetr_coco" / "annotations" / "instances_val2017.json"
    val_dir = data_dir / "valid"
    if not val_ann_path.exists():
        raise FileNotFoundError(f"Validation annotation file not found: {val_ann_path}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation image folder not found: {val_dir}")

    image_info, anns_by_img = _load_val_annotations(val_ann_path)
    image_ids = sorted(image_info.keys())
    targets = _build_targets(image_ids, anns_by_img)

    predictions, preds_by_img = _run_validation_predictions(args, image_info)
    metric_predictions = _prediction_dict_for_metrics(predictions, image_ids)

    metrics = calculate_metrics(metric_predictions, targets, score_threshold=args.score_thr)
    map_metrics = compute_map_metrics(metric_predictions, targets)
    metrics.update(map_metrics)

    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    plot_per_class_metrics(metrics["per_class_metrics"], output_dir / "per_class_metrics.png")
    plot_prediction_diagnostics(
        metric_predictions,
        targets,
        output_dir / "prediction_diagnostics.png",
        score_threshold=args.score_thr,
    )

    _render_validation_predictions(
        val_dir=val_dir,
        image_info=image_info,
        anns_by_img=anns_by_img,
        preds_by_img=preds_by_img,
        save_dir=output_dir / "visualizations",
        score_thr=args.score_thr,
        max_samples=args.max_visualizations,
    )

    with (output_dir / "val_pred.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    task2_predictions = create_task2_predictions(predictions, image_ids=image_ids)
    task2_gt = _build_task2_ground_truth(image_ids, anns_by_img)
    task2_pred_df = pd.DataFrame(task2_predictions)
    task2_gt_df = pd.DataFrame(task2_gt)
    task2_eval_df = task2_gt_df.merge(task2_pred_df, on="image_id", how="left")
    task2_eval_df["pred_label"] = task2_eval_df["pred_label"].fillna(-1).astype(int)
    task2_eval_df["correct"] = task2_eval_df["gt_label"] == task2_eval_df["pred_label"]
    task2_accuracy = float(task2_eval_df["correct"].mean()) if len(task2_eval_df) > 0 else 0.0
    task2_eval_df.to_csv(output_dir / "task2_eval.csv", index=False)

    with (output_dir / "per_class_metrics.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(format_per_class_metrics(metrics["per_class_metrics"])))

    summary = {
        "num_checkpoints": len(args.checkpoint),
        "score_threshold": float(args.score_thr),
        "tta_scales": [float(scale) for scale in args.tta_scales],
        "tta_horizontal_flip": bool(args.tta_horizontal_flip),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mAP": float(metrics["mAP"]),
        "AP50": float(metrics["AP50"]),
        "AP75": float(metrics["AP75"]),
        "task2_accuracy": task2_accuracy,
        "num_validation_images": len(image_ids),
        "num_exported_predictions": len(predictions),
    }

    if args.log_path is not None:
        log_path = Path(args.log_path).resolve()
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        _plot_curves_from_text_log(log_path, output_dir / "training_curves.png")
        parsed_log = _parse_text_log(log_path)
        if parsed_log["epochs"]:
            best_idx = int(np.nanargmax(parsed_log["mAP"]))
            summary["log_epochs"] = parsed_log["epochs"]
            summary["best_log_epoch_by_mAP"] = int(parsed_log["epochs"][best_idx])
            summary["best_log_mAP"] = float(parsed_log["mAP"][best_idx])
            summary["best_log_AP50"] = float(parsed_log["AP50"][best_idx])
            summary["best_log_AP75"] = float(parsed_log["AP75"][best_idx])

    with (output_dir / "report_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved report artifacts to {output_dir}")
    print(json.dumps(summary, indent=2))
    return summary


def _load_prediction_json(pred_json_path: Path):
    with pred_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_json_train_log(log_path: Path):
    # MMEngine JSON logs are one JSON object per line, not one large JSON array.
    train_loss_by_epoch = defaultdict(list)
    val_loss_by_epoch = {}
    map_by_epoch = {}
    ap50_by_epoch = {}
    ap75_by_epoch = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = row.get("epoch")
            if epoch is None:
                continue
            epoch = int(epoch)

            if "loss" in row and isinstance(row["loss"], (int, float)):
                train_loss_by_epoch[epoch].append(float(row["loss"]))
            if "loss_val" in row and isinstance(row["loss_val"], (int, float)):
                val_loss_by_epoch[epoch] = float(row["loss_val"])
            if "coco/bbox_mAP" in row:
                map_by_epoch[epoch] = float(row["coco/bbox_mAP"])
            if "coco/bbox_mAP_50" in row:
                ap50_by_epoch[epoch] = float(row["coco/bbox_mAP_50"])
            if "coco/bbox_mAP_75" in row:
                ap75_by_epoch[epoch] = float(row["coco/bbox_mAP_75"])

    epochs = sorted(set(train_loss_by_epoch) | set(val_loss_by_epoch) | set(map_by_epoch) | set(ap50_by_epoch) | set(ap75_by_epoch))
    return {
        "epochs": epochs,
        "train_loss": [float(np.mean(train_loss_by_epoch[e])) if e in train_loss_by_epoch else np.nan for e in epochs],
        "val_loss": [val_loss_by_epoch.get(e, np.nan) for e in epochs],
        "mAP": [map_by_epoch.get(e, np.nan) for e in epochs],
        "AP50": [ap50_by_epoch.get(e, np.nan) for e in epochs],
        "AP75": [ap75_by_epoch.get(e, np.nan) for e in epochs],
    }


def _parse_rich_text_train_log(log_path: Path):
    # Fall back to the custom rich text training log format if no JSON epochs are
    # found in the file.
    parsed = _parse_text_log(log_path)
    start_epoch_pattern = re.compile(r"Starting Epoch\s+(\d+)/")
    scalar_pattern = re.compile(r"^\s*(Loss|mAP|AP50|AP75)\s*:\s*([0-9.]+)")

    val_loss_by_epoch = {}
    current_epoch = None
    in_validation_block = False

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            start_epoch_match = start_epoch_pattern.search(line)
            if start_epoch_match:
                current_epoch = int(start_epoch_match.group(1))
                in_validation_block = False
                continue

            if "Validation Metrics:" in line:
                in_validation_block = True
                continue

            if not in_validation_block or current_epoch is None:
                continue

            scalar_match = scalar_pattern.search(line)
            if not scalar_match:
                continue

            if scalar_match.group(1) == "Loss":
                val_loss_by_epoch[current_epoch] = float(scalar_match.group(2))

        parsed["val_loss"] = [val_loss_by_epoch.get(epoch, np.nan) for epoch in parsed["epochs"]]
    return parsed


def _parse_training_log(log_path: Path):
    parsed = _parse_json_train_log(log_path)
    if parsed["epochs"]:
        return parsed
    return _parse_rich_text_train_log(log_path)


def _plot_training_curves_from_log(log_path: Path, save_path: Path):
    parsed = _parse_training_log(log_path)
    epochs = parsed["epochs"]

    if not epochs:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No epoch metrics found in train log", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].plot(epochs, parsed["train_loss"], "b-o", label="Train Loss")
    if "val_loss" in parsed and any(not np.isnan(v) for v in parsed["val_loss"]):
        axes[0].plot(epochs, parsed["val_loss"], "k--o", label="Val Loss")
    axes[0].set_title("Training Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, parsed["mAP"], "m-o", label="mAP")
    axes[1].plot(epochs, parsed["AP50"], "c-o", label="AP50")
    axes[1].plot(epochs, parsed["AP75"], "y-o", label="AP75")
    axes[1].set_title("Validation AP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AP")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _load_validation_log(log_path: Path):
    with log_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_log_path"] = str(log_path.resolve())
    return payload


def _normalize_per_class_metrics(per_class_metrics):
    return {int(class_id): metrics for class_id, metrics in per_class_metrics.items()}


def _collect_validation_logs(args):
    # Preserve the user-provided experiment group so the final comparison plot
    # can separate plain, flip, scale, and combined TTA runs.
    grouped_logs = []
    group_specs = [
        ("plain", args.validation_plain),
        ("flip", args.validation_flip),
        ("scale", args.validation_scale),
        ("both", args.validation_both),
    ]

    for group_name, paths in group_specs:
        if not paths:
            continue
        for path in paths:
            grouped_logs.append((group_name, Path(path).resolve()))

    if not grouped_logs:
        raise ValueError(
            "At least one validation log is required. Use --validation-plain, --validation-flip, --validation-scale, or --validation-both."
        )
    return grouped_logs


def _compute_run_metrics(run_info):
    # Reuse metrics embedded in validation logs when available; otherwise rebuild
    # them from the saved predictions and annotation paths.
    if run_info.get("confusion_matrix") is not None and run_info.get("per_class_metrics") is not None:
        return {
            "summary": run_info.get("summary", {}),
            "confusion_matrix": np.array(run_info["confusion_matrix"]),
            "per_class_metrics": _normalize_per_class_metrics(run_info["per_class_metrics"]),
        }

    ann_path = Path(run_info["ann_path"]).resolve()
    pred_json_path = Path(run_info["pred_json_path"]).resolve()
    image_info, anns_by_img = _load_val_annotations(ann_path)
    image_ids = sorted(image_info.keys())
    targets = _build_targets(image_ids, anns_by_img)
    predictions = _load_prediction_json(pred_json_path)
    metric_predictions = _prediction_dict_for_metrics(predictions, image_ids)
    score_thr = float(run_info.get("score_threshold", run_info.get("summary", {}).get("score_threshold", 0.5)))

    metrics = calculate_metrics(metric_predictions, targets, score_threshold=score_thr)
    metrics.update(compute_map_metrics(metric_predictions, targets))
    return {
        "summary": {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "mAP": float(metrics["mAP"]),
            "AP50": float(metrics["AP50"]),
            "AP75": float(metrics["AP75"]),
            "score_threshold": score_thr,
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": metrics["per_class_metrics"],
    }


def _select_best_run(runs):
    # Choose the best report run by the primary leaderboard-style metric first,
    # then use AP50 and F1 only as deterministic tie-breakers.
    return max(
        runs,
        key=lambda run: (
            float(run["summary"].get("mAP", 0.0)),
            float(run["summary"].get("AP50", 0.0)),
            float(run["summary"].get("f1", 0.0)),
        ),
    )


def _plot_tta_analysis(runs, save_path: Path):
    labels = [run["run_name"] for run in runs]
    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(labels) * 2.2), 10))
    series = [
        ("mAP", [float(run["summary"].get("mAP", 0.0)) for run in runs], "#7c3aed"),
        ("AP50", [float(run["summary"].get("AP50", 0.0)) for run in runs], "#06b6d4"),
        ("AP75", [float(run["summary"].get("AP75", 0.0)) for run in runs], "#f59e0b"),
        ("Task2", [float(run["summary"].get("task2_accuracy", 0.0)) for run in runs], "#16a34a"),
    ]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for (name, values, color), offset in zip(series, offsets):
        axes[0].bar(x + offset, values, width=width, label=name, color=color)

    axes[0].set_title("Validation / TTA Comparison")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].plot(labels, [float(run["summary"].get("precision", 0.0)) for run in runs], "o-", label="Precision")
    axes[1].plot(labels, [float(run["summary"].get("recall", 0.0)) for run in runs], "o-", label="Recall")
    axes[1].plot(labels, [float(run["summary"].get("f1", 0.0)) for run in runs], "o-", label="F1")
    axes[1].set_title("Detection Metrics by Validation Run")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _build_case_study_rows(run_info, top_k):
    ann_path = Path(run_info["ann_path"]).resolve()
    val_dir = Path(run_info["val_dir"]).resolve()
    pred_json_path = Path(run_info["pred_json_path"]).resolve()
    score_thr = float(run_info.get("score_threshold", run_info["summary"].get("score_threshold", 0.5)))

    image_info, anns_by_img = _load_val_annotations(ann_path)
    predictions = _load_prediction_json(pred_json_path)
    preds_by_img = defaultdict(list)
    for pred in predictions:
        if float(pred["score"]) >= score_thr:
            preds_by_img[int(pred["image_id"])].append(pred)

    rows = []
    for image_id in sorted(image_info.keys()):
        gt_anns = sorted(anns_by_img.get(image_id, []), key=lambda ann: ann["bbox"][0])
        pred_anns = sorted(preds_by_img.get(image_id, []), key=lambda pred: pred["bbox"][0])
        gt_label = int("".join(str(int(ann["category_id"]) % 10) for ann in gt_anns)) if gt_anns else -1
        pred_label = int("".join(str(int(pred["category_id"]) % 10) for pred in pred_anns)) if pred_anns else -1
        # Prioritize full-number mistakes and then images with the wrong digit count.
        error_score = (0 if gt_label == pred_label else 10) + abs(len(gt_anns) - len(pred_anns))
        if error_score <= 0:
            continue
        rows.append(
            {
                "image_id": image_id,
                "image_path": val_dir / image_info[image_id],
                "gt_anns": gt_anns,
                "pred_anns": pred_anns,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "error_score": error_score,
            }
        )

    rows.sort(key=lambda row: (-row["error_score"], row["image_id"]))
    return rows[:top_k]


def _plot_case_study(run_info, save_path: Path, top_k: int):
    cases = _build_case_study_rows(run_info, top_k=top_k)
    if not cases:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No obvious validation failure cases found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return

    cols = 2
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, case in zip(axes.flat, cases):
        image = Image.open(case["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)

        for ann in case["gt_anns"]:
            x1, y1, x2, y2 = _xywh_to_xyxy(ann["bbox"])
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)

        for pred in case["pred_anns"]:
            x, y, w, h = pred["bbox"]
            draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

        ax.imshow(image)
        ax.set_title(f"img {case['image_id']} | gt={case['gt_label']} pred={case['pred_label']}", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Case Study: {run_info['run_name']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def generate_report(train_log, validation_logs, output_dir, top_case_studies):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all validation logs, normalize their metrics, and rank them before
    # creating figures from the best run.
    runs = []
    for input_group, log_path in validation_logs:
        run = _load_validation_log(log_path)
        computed = _compute_run_metrics(run)
        run["summary"] = {**run.get("summary", {}), **computed["summary"]}
        run["confusion_matrix"] = computed["confusion_matrix"].tolist()
        run["per_class_metrics"] = computed["per_class_metrics"]
        run["input_group"] = input_group
        if not run.get("tta_mode"):
            run["tta_mode"] = input_group
        if run.get("run_name"):
            run["run_name"] = f"{input_group}:{run['run_name']}"
        else:
            run["run_name"] = input_group
        runs.append(run)

    if not runs:
        raise RuntimeError("At least one validation log is required.")

    runs.sort(key=lambda run: (-float(run["summary"].get("mAP", 0.0)), run["run_name"]))
    best_run = _select_best_run(runs)

    if train_log is not None:
        _plot_training_curves_from_log(train_log, output_dir / "training_curves.png")

    plot_confusion_matrix(np.array(best_run["confusion_matrix"]), output_dir / "confusion_matrix.png")
    plot_per_class_metrics(_normalize_per_class_metrics(best_run["per_class_metrics"]), output_dir / "per_class_metrics.png")

    ann_path = Path(best_run["ann_path"]).resolve()
    image_info, anns_by_img = _load_val_annotations(ann_path)
    image_ids = sorted(image_info.keys())
    targets = _build_targets(image_ids, anns_by_img)
    predictions = _load_prediction_json(Path(best_run["pred_json_path"]).resolve())
    metric_predictions = _prediction_dict_for_metrics(predictions, image_ids)
    score_thr = float(best_run.get("score_threshold", best_run["summary"].get("score_threshold", 0.5)))
    plot_prediction_diagnostics(metric_predictions, targets, output_dir / "prediction_diagnostics.png", score_threshold=score_thr)

    _plot_tta_analysis(runs, output_dir / "tta_analysis.png")
    _plot_case_study(best_run, output_dir / "case_study.png", top_k=top_case_studies)

    ranking_rows = []
    for run in runs:
        ranking_rows.append(
            {
                "run_name": run["run_name"],
                "tta_mode": run.get("tta_mode", "unknown"),
                "num_checkpoints": int(run.get("num_checkpoints", 0)),
                "mAP": float(run["summary"].get("mAP", 0.0)),
                "AP50": float(run["summary"].get("AP50", 0.0)),
                "AP75": float(run["summary"].get("AP75", 0.0)),
                "f1": float(run["summary"].get("f1", 0.0)),
                "task2_accuracy": float(run["summary"].get("task2_accuracy", 0.0)),
                "log_path": run["_log_path"],
            }
        )

    pd.DataFrame(ranking_rows).to_csv(output_dir / "tta_ranking.csv", index=False)
    with (output_dir / "per_class_metrics.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(format_per_class_metrics(_normalize_per_class_metrics(best_run["per_class_metrics"]))))

    summary = {
        "best_run_name": best_run["run_name"],
        "best_run_log": best_run["_log_path"],
        "best_run_metrics": best_run["summary"],
        "num_validation_runs": len(runs),
        "train_log": str(train_log.resolve()) if train_log is not None else None,
    }
    with (output_dir / "report_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved report artifacts to {output_dir}")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    train_log = Path(args.train_log).resolve() if args.train_log else None
    validation_logs = _collect_validation_logs(args)
    output_dir = Path(args.output_dir).resolve()
    generate_report(train_log, validation_logs, output_dir, top_case_studies=args.top_case_studies)


if __name__ == "__main__":
    main()
