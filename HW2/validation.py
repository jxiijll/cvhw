import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from inference import (
    _collect_predictions_for_image,
    _direct_predictions,
    _image_id_from_filename,
    _prepare_tta_views,
    _resolve_device,
    create_task2_predictions,
)
from report import (
    _build_task2_ground_truth,
    _build_targets,
    _load_val_annotations,
    _prediction_dict_for_metrics,
    _render_validation_predictions,
)
from utils import (
    calculate_metrics,
    compute_map_metrics,
    format_per_class_metrics,
)


# Run validation inference and save metrics, visualizations, and reusable logs.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Co-DETR validation inference with optional ensemble/TTA and export a validation log."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to MMDetection config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="One or more trained checkpoint files. Multiple checkpoints enable ensemble inference.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./nycu-hw2-data",
        help="Dataset root containing valid/ and codetr_coco/annotations/instances_val2017.json.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Optional explicit validation image folder path. Overrides --data-dir/valid.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./work_dirs/co_detr_r50_svhn_validation",
        help="Directory where validation artifacts and logs are saved.",
    )
    parser.add_argument("--score-thr", type=float, default=0.5, help="Score threshold for exported predictions.")
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Scale factors used for validation-time augmentation.",
    )
    parser.add_argument(
        "--tta-horizontal-flip",
        action="store_true",
        help="Enable horizontal-flip validation-time augmentation.",
    )
    parser.add_argument(
        "--ensemble-iou-thr",
        type=float,
        default=0.55,
        help="IoU threshold used to merge boxes from ensemble/TTA predictions.",
    )
    parser.add_argument(
        "--pre-merge-score-thr",
        type=float,
        default=0.05,
        help="Filter low-confidence boxes before ensemble/TTA merging.",
    )
    parser.add_argument("--device", type=str, default=None, help="Inference device string, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Optional path to a cloned repo containing projects/CO-DETR (e.g. external/mmdetection).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name. Default auto-infers from ensemble/TTA setting.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=8,
        help="How many validation examples to render as quick qualitative artifacts.",
    )
    return parser.parse_args()


def _sanitize_token(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return text.strip("-") or "run"


def _format_scale(scale: float) -> str:
    return str(scale).replace(".", "p")


def _infer_tta_mode(scales, use_flip: bool) -> str:
    use_scale = list(scales) != [1.0]
    if use_scale and use_flip:
        return "scale_flip"
    if use_scale:
        return "scale"
    if use_flip:
        return "flip"
    return "none"


def _build_run_name(args) -> str:
    if args.run_name:
        return _sanitize_token(args.run_name)

    # Encode the ensemble size and TTA mode in output filenames so multiple
    # validation experiments can share one output directory safely.
    tta_mode = _infer_tta_mode(args.tta_scales, args.tta_horizontal_flip)
    ckpt_token = f"ens{len(args.checkpoint)}"
    if tta_mode == "none":
        tta_token = "plain"
    elif tta_mode == "flip":
        tta_token = "flip"
    elif tta_mode == "scale":
        tta_token = "scale-" + "-".join(_format_scale(scale) for scale in args.tta_scales)
    else:
        tta_token = "both-" + "-".join(_format_scale(scale) for scale in args.tta_scales)
    return _sanitize_token(f"validation_{ckpt_token}_{tta_token}")


def _prepare_environment(args):
    # Some local setups keep MMDetection/CO-DETR inside external/, so optionally
    # add that repo before importing mmdet modules.
    if args.repo_root is not None:
        repo_root = str(Path(args.repo_root).resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    try:
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise ImportError(
            "MMDetection is not installed. Install the Co-DETR stack first before running validation.py."
        ) from exc

    register_all_modules(init_default_scope=True)


def _resolve_paths(args):
    data_dir = Path(args.data_dir).resolve()
    val_dir = Path(args.val_dir).resolve() if args.val_dir else (data_dir / "valid")
    ann_path = data_dir / "codetr_coco" / "annotations" / "instances_val2017.json"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation image folder not found: {val_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Validation annotation file not found: {ann_path}")
    return data_dir, val_dir, ann_path, output_dir


def _load_models(args):
    from mmdet.apis import init_detector

    device = _resolve_device(args.device)
    checkpoint_paths = [str(Path(path).resolve()) for path in args.checkpoint]
    models = [init_detector(args.config, checkpoint_path, device=device) for checkpoint_path in checkpoint_paths]

    print(f"Using {len(models)} checkpoint(s) on {device}:")
    for checkpoint_path in checkpoint_paths:
        print(f"  - {checkpoint_path}")
    print(
        "Validation TTA views per image: "
        f"{len(args.tta_scales) * (2 if args.tta_horizontal_flip else 1)} "
        f"(scales={args.tta_scales}, hflip={args.tta_horizontal_flip})"
    )
    return models, checkpoint_paths, device


def _run_validation_predictions(args, image_info, val_dir, models):
    from mmdet.apis import inference_detector
    import mmcv

    predictions = []
    preds_by_img = defaultdict(list)
    image_ids = []
    use_single_view = len(models) == 1 and not args.tta_horizontal_flip and args.tta_scales == [1.0]

    # Store all merged predictions for visualization, but only export boxes that
    # pass the final score threshold for metrics and Task 2 evaluation.
    for image_id in tqdm(sorted(image_info.keys()), desc="Running validation inference"):
        image_path = val_dir / image_info[image_id]
        if not image_path.exists():
            continue
        image_ids.append(int(image_id))

        if use_single_view:
            # Direct inference is equivalent to the final submission path without
            # TTA, but is cheaper than building TTA views and fusing boxes.
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
    return predictions, image_ids, preds_by_img


def _write_validation_log(log_path: Path, payload: dict):
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    _prepare_environment(args)
    data_dir, val_dir, ann_path, output_dir = _resolve_paths(args)
    run_name = _build_run_name(args)

    image_info, anns_by_img = _load_val_annotations(ann_path)
    models, checkpoint_paths, device = _load_models(args)
    predictions, image_ids, preds_by_img = _run_validation_predictions(args, image_info, val_dir, models)

    targets = _build_targets(image_ids, anns_by_img)
    metric_predictions = _prediction_dict_for_metrics(predictions, image_ids)
    metrics = calculate_metrics(metric_predictions, targets, score_threshold=args.score_thr)
    metrics.update(compute_map_metrics(metric_predictions, targets))

    task2_predictions = create_task2_predictions(predictions, image_ids=image_ids)
    task2_gt = _build_task2_ground_truth(image_ids, anns_by_img)
    # Task 2 accuracy checks whether the full left-to-right digit string matches.
    task2_eval_df = pd.DataFrame(task2_gt).merge(pd.DataFrame(task2_predictions), on="image_id", how="left")
    task2_eval_df["pred_label"] = task2_eval_df["pred_label"].fillna(-1).astype(int)
    task2_eval_df["correct"] = task2_eval_df["gt_label"] == task2_eval_df["pred_label"]
    task2_accuracy = float(task2_eval_df["correct"].mean()) if len(task2_eval_df) > 0 else 0.0

    pred_json_path = output_dir / f"val_pred_{run_name}.json"
    pred_csv_path = output_dir / f"val_pred_{run_name}.csv"
    task2_eval_path = output_dir / f"task2_eval_{run_name}.csv"
    per_class_path = output_dir / f"per_class_metrics_{run_name}.txt"
    vis_dir = output_dir / f"visualizations_{run_name}"
    log_path = output_dir / f"{run_name}.log"

    with pred_json_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)
    pd.DataFrame(task2_predictions).to_csv(pred_csv_path, index=False)
    task2_eval_df.to_csv(task2_eval_path, index=False)
    with per_class_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(format_per_class_metrics(metrics["per_class_metrics"])))

    _render_validation_predictions(
        val_dir=val_dir,
        image_info=image_info,
        anns_by_img=anns_by_img,
        preds_by_img=preds_by_img,
        save_dir=vis_dir,
        score_thr=args.score_thr,
        max_samples=args.max_visualizations,
    )

    summary = {
        # Keep the top-level summary compact so report.py can compare many runs.
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mAP": float(metrics["mAP"]),
        "AP50": float(metrics["AP50"]),
        "AP75": float(metrics["AP75"]),
        "task2_accuracy": task2_accuracy,
        "score_threshold": float(args.score_thr),
        "num_validation_images": len(image_ids),
        "num_exported_predictions": len(predictions),
    }

    payload = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "checkpoints": checkpoint_paths,
        "num_checkpoints": len(checkpoint_paths),
        "device": device,
        "data_dir": str(data_dir),
        "val_dir": str(val_dir),
        "ann_path": str(ann_path),
        "pred_json_path": str(pred_json_path),
        "pred_csv_path": str(pred_csv_path),
        "task2_eval_path": str(task2_eval_path),
        "visualizations_dir": str(vis_dir),
        "per_class_metrics_path": str(per_class_path),
        "score_threshold": float(args.score_thr),
        "tta_scales": [float(scale) for scale in args.tta_scales],
        "tta_horizontal_flip": bool(args.tta_horizontal_flip),
        "tta_mode": _infer_tta_mode(args.tta_scales, args.tta_horizontal_flip),
        "ensemble_iou_thr": float(args.ensemble_iou_thr),
        "pre_merge_score_thr": float(args.pre_merge_score_thr),
        "summary": summary,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "per_class_metrics": {str(k): v for k, v in metrics["per_class_metrics"].items()},
    }
    _write_validation_log(log_path, payload)

    print(f"Saved validation predictions: {pred_json_path}")
    print(f"Saved validation Task2 CSV: {pred_csv_path}")
    print(f"Saved validation log: {log_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
