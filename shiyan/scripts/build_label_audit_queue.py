"""Build a targeted, read-only manual label-audit queue from fixed predictions."""

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

from shiyan.scripts.analyze_errors import _match_image  # noqa: E402
from shiyan.src.inference.labels import CLASS_NAMES  # noqa: E402
from shiyan.src.inference.runner import read_image_list  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


def _parse_target_classes(values: list[str]) -> tuple[int, ...]:
    category_ids = tuple(dict.fromkeys(int(value) for value in values))
    if not category_ids or any(not 0 <= value < len(CLASS_NAMES) for value in category_ids):
        raise ValueError("target classes must be valid category IDs")
    return category_ids


def _draw_box(image, bbox: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _render_target_image(
    image_path: Path,
    match: dict[str, object],
    target_classes: set[int],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {image_path}")
    predictions = match["predictions"]
    matched_prediction = match["matched_prediction"]
    for prediction, matched in zip(predictions, matched_prediction):
        category_id = int(prediction["category_id"])
        if category_id not in target_classes:
            continue
        color = (0, 200, 0) if matched else (0, 165, 255)
        label = f"P {CLASS_NAMES[category_id]} {float(prediction['score']):.3f}"
        _draw_box(image, tuple(float(value) for value in prediction["bbox"]), color, label)
    for (category_id, bbox), matched in zip(match["ground_truth"], match["matched_truth"]):
        if category_id in target_classes and not matched:
            _draw_box(image, bbox, (0, 0, 255), f"FN {CLASS_NAMES[category_id]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"unable to write audit visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-class", action="append", default=["24", "3"])
    parser.add_argument("--top-images", type=int, default=30)
    args = parser.parse_args()
    if args.top_images <= 0:
        raise ValueError("top-images must be positive")
    target_classes = set(_parse_target_classes(args.target_class))

    image_paths = read_image_list(args.image_list)
    document = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    validate_result_document(document, [path.name for path in image_paths])
    entries = {entry["file_name"]: entry for entry in document["images"]}
    class_fp: Counter[int] = Counter()
    class_fn: Counter[int] = Counter()
    rows: list[dict[str, object]] = []
    matches: dict[str, dict[str, object]] = {}

    for image_path in image_paths:
        match = _match_image(image_path, entries[image_path.name])
        matches[image_path.name] = match
        target_fp = [category_id for category_id in match["false_positive_classes"] if category_id in target_classes]
        target_fn = [category_id for category_id in match["false_negative_classes"] if category_id in target_classes]
        if not target_fp and not target_fn:
            continue
        class_fp.update(target_fp)
        class_fn.update(target_fn)
        all_fp = len(match["false_positive_classes"])
        all_fn = len(match["false_negative_classes"])
        priority = 3 * len(target_fp) + 2 * len(target_fn) + (1 if all_fp + all_fn else 0)
        rows.append(
            {
                "file_name": image_path.name,
                "image_path": str(image_path),
                "target_classes": ",".join(CLASS_NAMES[index] for index in sorted(target_classes)),
                "target_fp": len(target_fp),
                "target_fn": len(target_fn),
                "target_errors": len(target_fp) + len(target_fn),
                "target_fp_classes": ",".join(CLASS_NAMES[index] for index in target_fp),
                "target_fn_classes": ",".join(CLASS_NAMES[index] for index in target_fn),
                "all_fp": all_fp,
                "all_fn": all_fn,
                "priority": priority,
                "review_status": "pending",
                "decision": "",
                "notes": "",
            }
        )

    rows.sort(key=lambda row: (int(row["priority"]), int(row["target_errors"]), int(row["all_fp"]) + int(row["all_fn"])), reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    fieldnames = [
        "rank",
        "file_name",
        "image_path",
        "target_classes",
        "target_fp",
        "target_fn",
        "target_errors",
        "target_fp_classes",
        "target_fn_classes",
        "all_fp",
        "all_fn",
        "priority",
        "review_status",
        "decision",
        "notes",
    ]
    with (output_dir / "review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rendered = min(args.top_images, len(rows))
    for row in rows[:rendered]:
        _render_target_image(
            Path(row["image_path"]),
            matches[row["file_name"]],
            target_classes,
            output_dir / "visualizations" / f"{int(row['rank']):02d}_{row['file_name']}",
        )

    summary = {
        "evaluation_scope": "internal_fixed_prediction_label_audit_queue",
        "image_count": len(image_paths),
        "queue_image_count": len(rows),
        "top_images_rendered": rendered,
        "target_classes": {str(category_id): CLASS_NAMES[category_id] for category_id in sorted(target_classes)},
        "false_positive_by_class": {CLASS_NAMES[index]: class_fp[index] for index in sorted(target_classes)},
        "false_negative_by_class": {CLASS_NAMES[index]: class_fn[index] for index in sorted(target_classes)},
        "original_labels_modified": False,
        "review_status_values": ["pending", "confirmed_label_issue", "model_error", "ambiguous", "reviewed_no_change"],
        "outputs": ["review_queue.csv", "visualizations/", "audit_summary.json"],
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"images={len(image_paths)}")
    print(f"queue_images={len(rows)}")
    print(f"visualizations={rendered}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
