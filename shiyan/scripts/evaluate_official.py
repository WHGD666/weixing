"""Evaluate local result.json with the competition metric contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.src.inference.labels import (  # noqa: E402
    AIRCRAFT_CLASS_IDS,
    CLASS_NAMES,
    SHIP_CLASS_IDS,
    VEHICLE_CLASS_IDS,
    class_group,
)
from shiyan.src.inference.runner import read_image_list  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_index = next((index for index, part in enumerate(parts) if part.lower() == "images"), None)
    if image_index is None:
        raise ValueError(f"cannot derive labels path from image path: {image_path}")
    return Path(*parts[:image_index], "labels", *parts[image_index + 1 :]).with_suffix(".txt")


def _read_ground_truth(image_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    label_path = _label_path(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"label file not found for {image_path}: {label_path}")
    ground_truth: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = raw_line.split()
        if len(values) != 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        category_id = int(values[0])
        xc, yc, box_width, box_height = [float(value) for value in values[1:]]
        x1 = max(0.0, (xc - box_width / 2.0) * width)
        y1 = max(0.0, (yc - box_height / 2.0) * height)
        x2 = min(float(width), (xc + box_width / 2.0) * width)
        y2 = min(float(height), (yc + box_height / 2.0) * height)
        ground_truth.append((category_id, (x1, y1, x2, y2)))
    return ground_truth


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _threshold(category_id: int) -> float:
    return 0.35 if category_id in VEHICLE_CLASS_IDS else 0.50


def _new_stats() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _metric_row(category_id: int, stats: dict[str, int]) -> dict[str, object]:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    support = tp + fn
    predicted = tp + fp
    return {
        "category_id": category_id,
        "category_name": CLASS_NAMES[category_id],
        "group": class_group(category_id),
        "iou_threshold": _threshold(category_id),
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / support if support else None,
        "fdr": fp / predicted if predicted else None,
    }


def _aggregate(stats: Iterable[dict[str, int]]) -> dict[str, object]:
    merged = _new_stats()
    for item in stats:
        for key in merged:
            merged[key] += item[key]
    tp, fp, fn = merged["tp"], merged["fp"], merged["fn"]
    support = tp + fn
    predicted = tp + fp
    return {
        **merged,
        "support": support,
        "recall": tp / support if support else None,
        "fdr": fp / predicted if predicted else None,
    }


def evaluate(prediction_path: Path, image_list_path: Path, timings_path: Path | None) -> dict[str, object]:
    image_paths = read_image_list(image_list_path)
    document = json.loads(prediction_path.read_text(encoding="utf-8"))
    validate_result_document(document, [path.name for path in image_paths])
    entries = {entry["file_name"]: entry for entry in document["images"]}
    class_stats = [_new_stats() for _ in CLASS_NAMES]

    for image_path in image_paths:
        entry = entries[image_path.name]
        width, height = int(entry["width"]), int(entry["height"])
        ground_truth = _read_ground_truth(image_path, width, height)
        matched = [False] * len(ground_truth)
        predictions = sorted(entry["objects"], key=lambda item: float(item["score"]), reverse=True)
        for prediction in predictions:
            category_id = int(prediction["category_id"])
            candidates = [
                index
                for index, (truth_category, _) in enumerate(ground_truth)
                if truth_category == category_id and not matched[index]
            ]
            best_index = None
            best_iou = 0.0
            prediction_box = tuple(float(value) for value in prediction["bbox"])
            for index in candidates:
                candidate_iou = _iou(prediction_box, ground_truth[index][1])
                if candidate_iou > best_iou:
                    best_index, best_iou = index, candidate_iou
            if best_index is not None and best_iou >= _threshold(category_id):
                matched[best_index] = True
                class_stats[category_id]["tp"] += 1
            else:
                class_stats[category_id]["fp"] += 1
        for index, (category_id, _) in enumerate(ground_truth):
            if not matched[index]:
                class_stats[category_id]["fn"] += 1

    per_class = [_metric_row(category_id, stats) for category_id, stats in enumerate(class_stats)]
    groups = {
        "ship": _aggregate(class_stats[index] for index in sorted(SHIP_CLASS_IDS)),
        "aircraft": _aggregate(class_stats[index] for index in sorted(AIRCRAFT_CLASS_IDS)),
        "vehicle": _aggregate(class_stats[index] for index in sorted(VEHICLE_CLASS_IDS)),
    }
    overall = _aggregate(class_stats)
    valid_group_recalls = [float(item["recall"]) for item in groups.values() if item["recall"] is not None]
    valid_group_fdrs = [float(item["fdr"]) for item in groups.values() if item["fdr"] is not None]

    timing_summary: dict[str, object] = {"available": False}
    if timings_path is not None:
        timing_document = json.loads(timings_path.read_text(encoding="utf-8"))
        timing_summary = {
            "available": True,
            "image_count": timing_document["image_count"],
            "total_seconds": timing_document["total_seconds"],
            "max_image_seconds": timing_document["max_image_seconds"],
            "pass_latency_20s": float(timing_document["max_image_seconds"]) <= 20.0,
        }

    return {
        "evaluation_scope": "internal_validation_only",
        "official_hidden_test": False,
        "metric_contract": "Metric Contract v0",
        "iou_thresholds": {"vehicle": 0.35, "ship_and_aircraft": 0.50},
        "image_count": len(image_paths),
        "overall": overall,
        "groups": groups,
        "group_mean": {
            "recall": sum(valid_group_recalls) / len(valid_group_recalls) if valid_group_recalls else None,
            "fdr": sum(valid_group_fdrs) / len(valid_group_fdrs) if valid_group_fdrs else None,
        },
        "per_class": per_class,
        "timing": timing_summary,
        "gates": {
            "recall_ge_0_85": overall["recall"] is not None and float(overall["recall"]) >= 0.85,
            "fdr_le_0_20": overall["fdr"] is not None and float(overall["fdr"]) <= 0.20,
            "latency_le_20s": timing_summary.get("pass_latency_20s"),
        },
    }


def _write_outputs(metrics: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "official_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["category_id", "category_name", "group", "iou_threshold", "support", "tp", "fp", "fn", "recall", "fdr"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    overall = metrics["overall"]
    timing = metrics["timing"]
    lines = [
        "# Internal Official-Metric Evaluation",
        "",
        "This is an internal validation result, not an official hidden-test score.",
        "",
        f"- Images: {metrics['image_count']}",
        f"- Overall TP / FP / FN: {overall['tp']} / {overall['fp']} / {overall['fn']}",
        f"- Overall Recall: {overall['recall']}",
        f"- Overall FDR: {overall['fdr']}",
        f"- Timing: {timing}",
        "",
        "| Group | Recall | FDR | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, item in metrics["groups"].items():
        lines.append(f"| {group} | {item['recall']} | {item['fdr']} | {item['tp']} | {item['fp']} | {item['fn']} |")
    (output_dir / "official_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timings")
    args = parser.parse_args()
    prediction_path = Path(args.predictions)
    timings_path = Path(args.timings) if args.timings else None
    metrics = evaluate(prediction_path, Path(args.image_list), timings_path)
    _write_outputs(metrics, Path(args.output_dir))
    print(json.dumps(metrics["gates"], ensure_ascii=False))
    print(f"metrics={Path(args.output_dir) / 'official_metrics.json'}")


if __name__ == "__main__":
    main()
