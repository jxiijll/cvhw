import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# Run Co-DETR inference and create both homework submission files.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Co-DETR inference on SVHN test images and export Task1/Task2 predictions."
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
        help="Dataset root containing test/ folder.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Optional explicit test image folder path. Overrides --data-dir/test.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./work_dirs/co_detr_r50_svhn",
        help="Where pred.json and pred.csv are saved.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.5,
        help="Score threshold for exported predictions.",
    )
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Scale factors used for test-time augmentation, e.g. --tta-scales 0.9 1.0 1.1.",
    )
    parser.add_argument(
        "--tta-horizontal-flip",
        action="store_true",
        help="Enable horizontal-flip test-time augmentation.",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device string, e.g. cuda:0 or cpu. Default auto-select.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Optional path to a cloned repo containing projects/CO-DETR (e.g., external/mmdetection).",
    )
    return parser.parse_args()


def category_id_to_digit(category_id: int) -> int:
    return int(category_id) % 10


def create_task2_predictions(predictions, image_ids):
    # Task 2 treats all detected digits in one image as a number read from left
    # to right, so the x coordinate of each box defines the digit order.
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        predictions_by_image[pred["image_id"]].append(pred)

    results = []
    for image_id in sorted(image_ids):
        preds = predictions_by_image.get(image_id, [])
        if len(preds) == 0:
            results.append({"image_id": image_id, "pred_label": -1})
            continue

        preds = sorted(preds, key=lambda item: item["bbox"][0])
        digits = [str(category_id_to_digit(item["category_id"])) for item in preds]
        pred_label = int("".join(digits)) if digits else -1
        results.append({"image_id": image_id, "pred_label": pred_label})

    return results


def _image_id_from_filename(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def _resolve_device(device_arg: str) -> str:
    import torch

    if device_arg:
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _xyxy_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _merge_boxes(boxes, scores, iou_thr: float):
    if not boxes:
        return []

    # Weighted box fusion: high-confidence boxes seed clusters, and overlapping
    # boxes vote for the final coordinates by confidence-weighted averaging.
    order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
    merged = []

    while order:
        seed_idx = order.pop(0)
        cluster = [seed_idx]
        remaining = []

        for idx in order:
            if _xyxy_iou(boxes[seed_idx], boxes[idx]) >= iou_thr:
                cluster.append(idx)
            else:
                remaining.append(idx)
        order = remaining

        weight_sum = sum(max(scores[idx], 1e-6) for idx in cluster)
        fused_box = [
            sum(boxes[idx][coord] * max(scores[idx], 1e-6) for idx in cluster) / weight_sum
            for coord in range(4)
        ]
        fused_score = sum(scores[idx] for idx in cluster) / len(cluster)
        merged.append((fused_box, fused_score))

    return merged


def _prepare_tta_views(image, scales, use_horizontal_flip):
    import mmcv
    import numpy as np

    orig_h, orig_w = image.shape[:2]
    views = []

    # Each view keeps enough metadata to map predictions back to the original
    # image coordinate system after resizing and optional horizontal flipping.
    for scale in scales:
        if scale <= 0:
            raise ValueError("--tta-scales values must be positive.")

        if abs(scale - 1.0) < 1e-8:
            scaled_image = image
        else:
            scaled_w = max(1, int(round(orig_w * scale)))
            scaled_h = max(1, int(round(orig_h * scale)))
            scaled_image = mmcv.imresize(image, (scaled_w, scaled_h))

        scaled_h, scaled_w = scaled_image.shape[:2]
        views.append(
            {
                "image": scaled_image,
                "scale_x": scaled_w / orig_w,
                "scale_y": scaled_h / orig_h,
                "flipped": False,
                "width": scaled_w,
            }
        )

        if use_horizontal_flip:
            views.append(
                {
                    "image": np.ascontiguousarray(scaled_image[:, ::-1, :]),
                    "scale_x": scaled_w / orig_w,
                    "scale_y": scaled_h / orig_h,
                    "flipped": True,
                    "width": scaled_w,
                }
            )

    return views


def _restore_boxes_to_original_space(bboxes, view):
    # Undo the TTA transform so all model outputs can be merged in one space.
    restored = []
    for bbox in bboxes:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if view["flipped"]:
            x1, x2 = view["width"] - x2, view["width"] - x1
        restored.append(
            [
                x1 / view["scale_x"],
                y1 / view["scale_y"],
                x2 / view["scale_x"],
                y2 / view["scale_y"],
            ]
        )
    return restored


def _direct_predictions(result, score_thr: float):
    predictions = []
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.detach().cpu().tolist()
    labels = pred_instances.labels.detach().cpu().tolist()
    scores = pred_instances.scores.detach().cpu().tolist()

    for bbox, label, score in zip(bboxes, labels, scores):
        if float(score) < score_thr:
            continue
        predictions.append((bbox, int(label), float(score)))

    predictions.sort(key=lambda item: (item[0][0], -item[2]))
    return predictions


def _collect_predictions_for_image(models, tta_views, iou_thr: float, pre_merge_score_thr: float):
    from mmdet.apis import inference_detector

    per_label_boxes = defaultdict(list)
    per_label_scores = defaultdict(list)

    # Run every model on every TTA view, then merge boxes class-by-class to avoid
    # fusing boxes from different digit categories.
    for model in models:
        for view in tta_views:
            result = inference_detector(model, view["image"])
            pred_instances = result.pred_instances

            restored_boxes = _restore_boxes_to_original_space(
                pred_instances.bboxes.detach().cpu().tolist(), view
            )
            labels = pred_instances.labels.detach().cpu().tolist()
            scores = pred_instances.scores.detach().cpu().tolist()

            for bbox, label, score in zip(restored_boxes, labels, scores):
                if float(score) < pre_merge_score_thr:
                    continue
                per_label_boxes[int(label)].append(bbox)
                per_label_scores[int(label)].append(float(score))

    merged_predictions = []
    for label in sorted(per_label_boxes):
        merged = _merge_boxes(per_label_boxes[label], per_label_scores[label], iou_thr=iou_thr)
        for bbox, score in merged:
            merged_predictions.append((bbox, label, score))

    merged_predictions.sort(key=lambda item: (item[0][0], -item[2]))
    return merged_predictions


def main() -> None:
    args = parse_args()
    import pandas as pd
    from tqdm import tqdm
    import mmcv

    if args.repo_root is not None:
        repo_root = str(Path(args.repo_root).resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    try:
        from mmdet.apis import inference_detector, init_detector
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise ImportError(
            "MMDetection is not installed. Install the Co-DETR stack first; see requirements-codetr.txt."
        ) from exc

    register_all_modules(init_default_scope=True)

    test_dir = Path(args.test_dir).resolve() if args.test_dir else (Path(args.data_dir).resolve() / "test")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The homework image filenames are numeric, so sorting by id keeps the output
    # deterministic and aligned with the expected submission order.
    image_paths = sorted(test_dir.glob("*.png"), key=lambda p: (_image_id_from_filename(p), p.name))
    if not image_paths:
        raise RuntimeError(f"No .png files found in {test_dir}")

    device = _resolve_device(args.device)
    checkpoint_paths = [str(Path(path).resolve()) for path in args.checkpoint]
    models = [init_detector(args.config, checkpoint_path, device=device) for checkpoint_path in checkpoint_paths]

    print(f"Using {len(models)} checkpoint(s) on {device}:")
    for checkpoint_path in checkpoint_paths:
        print(f"  - {checkpoint_path}")
    print(
        "TTA views per image: "
        f"{len(args.tta_scales) * (2 if args.tta_horizontal_flip else 1)} "
        f"(scales={args.tta_scales}, hflip={args.tta_horizontal_flip})"
    )

    task1_predictions = []
    all_image_ids = []
    use_legacy_single_view = len(models) == 1 and not args.tta_horizontal_flip and args.tta_scales == [1.0]

    for image_path in tqdm(image_paths, desc="Running Co-DETR inference"):
        image_id = _image_id_from_filename(image_path)
        if image_id < 0:
            continue
        all_image_ids.append(image_id)

        if use_legacy_single_view:
            # Keep the plain single-model path fast by skipping image reloading
            # and box fusion when no TTA or ensemble is requested.
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
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            # MMDetection labels are usually 0-based class indices; homework submission uses category_id 1..10.
            category_id = int(label) + 1

            task1_predictions.append(
                {
                    "image_id": int(image_id),
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                    "category_id": category_id,
                }
            )

    task1_predictions.sort(key=lambda item: (item["image_id"], item["bbox"][0], -item["score"]))
    task2_predictions = create_task2_predictions(task1_predictions, image_ids=all_image_ids)

    task1_path = output_dir / "pred.json"
    task2_path = output_dir / "pred.csv"

    with task1_path.open("w", encoding="utf-8") as f:
        json.dump(task1_predictions, f)

    pd.DataFrame(task2_predictions).to_csv(task2_path, index=False)

    print(f"Saved Task1 predictions: {task1_path}")
    print(f"Saved Task2 predictions: {task2_path}")


if __name__ == "__main__":
    main()
