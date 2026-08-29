"""Rank validation errors and render representative false-positive/negative cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.scripts.evaluate_official import (  # noqa: E402
    _iou,
    _read_ground_truth,
    _threshold,
)
from shiyan.src.inference.labels import CLASS_NAMES  # noqa: E402
from shiyan.src.inference.runner import read_image_list  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


def _match_image(image_path: Path, entry: dict[str, object]) -> dict[str, object]:
    width, height = int(entry["width"]), int(entry["height"])
    ground_truth = _read_ground_truth(image_path, width, height)
    predictions = sorted(entry["objects"], key=lambda item: float(item["score"]), reverse=True)
    matched_truth = [False] * len(ground_truth)
    matched_prediction = [False] * len(predictions)

    for prediction_index, prediction in enumerate(predictions):
        category_id = int(prediction["category_id"])
        prediction_box = tuple(float(value) for value in prediction["bbox"])
        candidates = [
            index
            for index, (truth_category, _) in enumerate(ground_truth)
            if truth_category == category_id and not matched_truth[index]
        ]
        best_index = None
        best_iou = 0.0
        for truth_index in candidates:
            candidate_iou = _iou(prediction_box, ground_truth[truth_index][1])
            if candidate_iou > best_iou:
                best_index, best_iou = truth_index, candidate_iou
        if best_index is not None and best_iou >= _threshold(category_id):
            matched_truth[best_index] = True
            matched_prediction[prediction_index] = True

    false_positive_classes = [
        int(prediction["category_id"])
        for prediction, matched in zip(predictions, matched_prediction)
        if not matched
    ]
    false_negative_classes = [
        category_id
        for (category_id, _), matched in zip(ground_truth, matched_truth)
        if not matched
    ]
    return {
        "ground_truth": ground_truth,
        "predictions": predictions,
        "matched_truth": matched_truth,
        "matched_prediction": matched_prediction,
        "false_positive_classes": false_positive_classes,
        "false_negative_classes": false_negative_classes,
    }


def _draw_box(image, bbox: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _render_error_image(image_path: Path, match: dict[str, object], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {image_path}")
    predictions = match["predictions"]
    matched_prediction = match["matched_prediction"]
    for prediction, matched in zip(predictions, matched_prediction):
        category_id = int(prediction["category_id"])
        color = (0, 200, 0) if matched else (0, 165, 255)
        label = f"P {CLASS_NAMES[category_id]} {float(prediction['score']):.3f}"
        _draw_box(image, tuple(float(value) for value in prediction["bbox"]), color, label)
    for (category_id, bbox), matched in zip(match["ground_truth"], match["matched_truth"]):
        if not matched:
            _draw_box(image, bbox, (0, 0, 255), f"FN {CLASS_NAMES[category_id]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"unable to write error visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-images", type=int, default=30)
    args = parser.parse_args()
    if args.top_images <= 0:
        raise ValueError("top-images must be positive")

    image_paths = read_image_list(args.image_list)
    document = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    validate_result_document(document, [path.name for path in image_paths])
    entries = {entry["file_name"]: entry for entry in document["images"]}
    class_fp = Counter()
    class_fn = Counter()
    image_rows: list[dict[str, object]] = []
    matches: dict[str, dict[str, object]] = {}

    for image_path in image_paths:
        match = _match_image(image_path, entries[image_path.name])
        matches[image_path.name] = match
        fp_classes = match["false_positive_classes"]
        fn_classes = match["false_negative_classes"]
        class_fp.update(fp_classes)
        class_fn.update(fn_classes)
        image_rows.append(
            {
                "file_name": image_path.name,
                "image_path": str(image_path),
                "fp": len(fp_classes),
                "fn": len(fn_classes),
                "errors": len(fp_classes) + len(fn_classes),
                "fp_classes": ",".join(CLASS_NAMES[index] for index in fp_classes),
                "fn_classes": ",".join(CLASS_NAMES[index] for index in fn_classes),
            }
        )

    image_rows.sort(key=lambda row: (int(row["errors"]), int(row["fp"]), int(row["fn"])), reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "error_images.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["file_name", "image_path", "fp", "fn", "errors", "fp_classes", "fn_classes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)

    class_rows = [
        {
            "category_id": category_id,
            "category_name": CLASS_NAMES[category_id],
            "false_positive": class_fp[category_id],
            "false_negative": class_fn[category_id],
        }
        for category_id in range(len(CLASS_NAMES))
    ]
    with (output_dir / "error_per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["category_id", "category_name", "false_positive", "false_negative"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(class_rows)

    top_rows = image_rows[: args.top_images]
    for row in top_rows:
        _render_error_image(
            Path(row["image_path"]),
            matches[row["file_name"]],
            output_dir / "visualizations" / row["file_name"],
        )
    summary = {
        "evaluation_scope": "internal_validation_error_analysis",
        "image_count": len(image_paths),
        "images_with_errors": sum(int(row["errors"]) > 0 for row in image_rows),
        "top_images_rendered": len(top_rows),
        "top_images": top_rows,
        "per_class": class_rows,
        "label_path_rule": "derived by replacing images with labels",
    }
    (output_dir / "error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"images={len(image_paths)}")
    print(f"images_with_errors={summary['images_with_errors']}")
    print(f"top_images_rendered={len(top_rows)}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
