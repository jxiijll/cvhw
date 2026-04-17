import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


# Convert the homework SVHN-style COCO files into the layout expected by MMDetection.


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _normalize_train_val(payload: Dict[str, Any]) -> Dict[str, Any]:
    # MMDetection expects every annotation file to contain categories and an
    # iscrowd flag, even though the homework labels are simple digit boxes.
    categories = payload.get("categories", [])
    if not categories:
        categories = [{"id": i + 1, "name": str(i), "supercategory": "digit"} for i in range(10)]

    for ann in payload.get("annotations", []):
        ann.setdefault("iscrowd", 0)

    return {
        "images": payload.get("images", []),
        "annotations": payload.get("annotations", []),
        "categories": categories,
    }


def _build_test_image_info(test_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    from PIL import Image

    # Test annotations do not contain boxes, so we export only image metadata.
    images = []
    for image_path in sorted(test_dir.glob("*.png")):
        stem = image_path.stem
        try:
            image_id = int(stem)
        except ValueError:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
            }
        )

    return {
        "images": images,
        "annotations": [],
        "categories": [{"id": i + 1, "name": str(i), "supercategory": "digit"} for i in range(10)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SVHN homework annotations to Co-DETR/MMDetection-compatible COCO files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./nycu-hw2-data",
        help="Dataset root containing train.json, valid.json, and train/ valid/ test folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <data-dir>/codetr_coco/annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    train_json = data_dir / "train.json"
    valid_json = data_dir / "valid.json"
    test_dir = data_dir / "test"

    if not train_json.exists():
        raise FileNotFoundError(f"Missing file: {train_json}")
    if not valid_json.exists():
        raise FileNotFoundError(f"Missing file: {valid_json}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing folder: {test_dir}")

    # Match the COCO filename convention used by the Co-DETR base configs.
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (data_dir / "codetr_coco" / "annotations")

    train_payload = _normalize_train_val(_load_json(train_json))
    valid_payload = _normalize_train_val(_load_json(valid_json))
    test_payload = _build_test_image_info(test_dir)

    train_out = output_dir / "instances_train2017.json"
    val_out = output_dir / "instances_val2017.json"
    test_out = output_dir / "image_info_test-dev2017.json"

    _write_json(train_out, train_payload)
    _write_json(val_out, valid_payload)
    _write_json(test_out, test_payload)

    print("Export finished.")
    print(f"Train annotations: {train_out}")
    print(f"Val annotations:   {val_out}")
    print(f"Test image info:   {test_out}")


if __name__ == "__main__":
    main()
