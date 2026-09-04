from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now, write_json
from .data_contract import parse_label_file
from .inference.labels import CLASS_NAMES, GROUP_CLASS_IDS, class_group
from .inference.postprocess import box_iou_xyxy
from .inference.schema import validate_result_document


def _new_stats() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _iou_threshold(category_id: int) -> float:
    return 0.35 if category_id == 24 else 0.50


def _aggregate(stats: Iterable[dict[str, int]]) -> dict[str, Any]:
    merged = _new_stats()
    for item in stats:
        for key in merged:
            merged[key] += item[key]
    support = merged["tp"] + merged["fn"]
    predicted = merged["tp"] + merged["fp"]
    return {
        **merged,
        "support": support,
        "predicted": predicted,
        "recall": merged["tp"] / support if support else None,
        "fdr": merged["fp"] / predicted if predicted else None,
    }


def _metric_row(category_id: int, stats: dict[str, int]) -> dict[str, Any]:
    return {
        "category_id": category_id,
        "category_name": CLASS_NAMES[category_id],
        "group": class_group(category_id),
        "iou_threshold": _iou_threshold(category_id),
        **_aggregate([stats]),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def _gates(recall: float | None, fdr: float | None, latency: bool | None) -> dict[str, bool | None]:
    return {
        "recall_ge_0_85": recall is not None and recall >= 0.85,
        "fdr_le_0_20": fdr is not None and fdr <= 0.20,
        "latency_le_20s": latency,
    }


def _read_ground_truth(
    label_path: Path, width: int, height: int
) -> list[tuple[int, tuple[float, float, float, float]]]:
    rows, errors = parse_label_file(label_path, len(CLASS_NAMES))
    if errors:
        raise ValueError("; ".join(errors))
    result: list[tuple[int, tuple[float, float, float, float]]] = []
    for row in rows:
        result.append(
            (
                row.class_id,
                (
                    max(0.0, (row.x_center - row.width / 2.0) * width),
                    max(0.0, (row.y_center - row.height / 2.0) * height),
                    min(float(width), (row.x_center + row.width / 2.0) * width),
                    min(float(height), (row.y_center + row.height / 2.0) * height),
                ),
            )
        )
    return result


def evaluate_protocol(
    *,
    prediction_path: Path,
    label_root: Path,
    expected_stems: Sequence[str],
    timings_path: Path | None,
    protocol_id: str,
) -> dict[str, Any]:
    document = json.loads(prediction_path.read_text(encoding="utf-8"))
    expected_names = [f"{stem}.jpg" for stem in expected_stems]
    actual_names = [str(entry["file_name"]) for entry in document.get("images", [])]
    if [Path(name).stem for name in actual_names] != list(expected_stems):
        raise ValueError("prediction order/coverage does not match the frozen validation split")
    validate_result_document(document, actual_names)
    entries = {Path(str(entry["file_name"])).stem: entry for entry in document["images"]}
    class_stats = [_new_stats() for _ in CLASS_NAMES]

    for stem in expected_stems:
        entry = entries[stem]
        width, height = int(entry["width"]), int(entry["height"])
        label_path = label_root / f"{stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"protocol label missing: {label_path}")
        truth = _read_ground_truth(label_path, width, height)
        matched = [False] * len(truth)
        predictions = sorted(entry["objects"], key=lambda item: float(item["score"]), reverse=True)
        for prediction in predictions:
            category_id = int(prediction["category_id"])
            prediction_box = tuple(float(value) for value in prediction["bbox"])
            best_index: int | None = None
            best_iou = 0.0
            for index, (truth_category, truth_box) in enumerate(truth):
                if matched[index] or truth_category != category_id:
                    continue
                value = box_iou_xyxy(prediction_box, truth_box)
                if value > best_iou:
                    best_iou, best_index = value, index
            if best_index is not None and best_iou >= _iou_threshold(category_id):
                matched[best_index] = True
                class_stats[category_id]["tp"] += 1
            else:
                class_stats[category_id]["fp"] += 1
        for index, (category_id, _box) in enumerate(truth):
            if not matched[index]:
                class_stats[category_id]["fn"] += 1

    per_class = [_metric_row(index, stats) for index, stats in enumerate(class_stats)]
    groups = {
        group: _aggregate(class_stats[index] for index in sorted(class_ids))
        for group, class_ids in GROUP_CLASS_IDS.items()
    }
    group_mean = {
        "recall": _mean(item["recall"] for item in groups.values()),
        "fdr": _mean(item["fdr"] for item in groups.values()),
    }
    class_macro_by_group = {
        group: {
            "recall": _mean(per_class[index]["recall"] for index in sorted(class_ids)),
            "fdr": _mean(per_class[index]["fdr"] for index in sorted(class_ids)),
        }
        for group, class_ids in GROUP_CLASS_IDS.items()
    }
    class_macro_mean = {
        "recall": _mean(item["recall"] for item in class_macro_by_group.values()),
        "fdr": _mean(item["fdr"] for item in class_macro_by_group.values()),
    }
    overall = _aggregate(class_stats)

    timing: dict[str, Any] = {"available": False, "pass_latency_20s": None}
    if timings_path is not None:
        timing_document = json.loads(timings_path.read_text(encoding="utf-8"))
        if int(timing_document["image_count"]) != len(expected_stems):
            raise ValueError("timing image_count does not match protocol")
        max_seconds = float(timing_document["max_image_seconds"])
        timing = {
            "available": True,
            "scope": "local_hardware_proxy",
            "image_count": int(timing_document["image_count"]),
            "total_seconds": float(timing_document["total_seconds"]),
            "average_image_seconds": float(timing_document["total_seconds"]) / len(expected_stems),
            "max_image_seconds": max_seconds,
            "pass_latency_20s": max_seconds <= 20.0,
        }

    return {
        "created_at_utc": utc_now(),
        "scope": "internal_validation_only",
        "protocol_id": protocol_id,
        "prediction_path": str(prediction_path.resolve()),
        "label_root": str(label_root.resolve()),
        "image_count": len(expected_stems),
        "matching": {"ship_iou": 0.50, "aircraft_iou": 0.50, "vehicle_iou": 0.35},
        "overall": overall,
        "groups": groups,
        "group_mean": group_mean,
        "class_macro_by_group": class_macro_by_group,
        "group_class_macro_mean": class_macro_mean,
        "per_class": per_class,
        "timing": timing,
        "gates": _gates(group_mean["recall"], group_mean["fdr"], timing["pass_latency_20s"]),
        "alternative_class_macro_gates": _gates(
            class_macro_mean["recall"], class_macro_mean["fdr"], timing["pass_latency_20s"]
        ),
    }


def write_protocol_outputs(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "official_metrics.json", metrics)
    with (output_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0]))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    lines = [
        f"# Internal metrics: {metrics['protocol_id']}",
        "",
        "This is an internal validation result, not an official hidden-test score.",
        "",
        f"- Images: {metrics['image_count']}",
        f"- Group-mean Recall: {metrics['group_mean']['recall']}",
        f"- Group-mean FDR: {metrics['group_mean']['fdr']}",
        f"- Gates: {metrics['gates']}",
        f"- Alternative class-macro Recall: {metrics['group_class_macro_mean']['recall']}",
        f"- Alternative class-macro FDR: {metrics['group_class_macro_mean']['fdr']}",
        f"- Timing: {metrics['timing']}",
        "",
        "| Group | Recall | FDR | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, item in metrics["groups"].items():
        lines.append(
            f"| {group} | {item['recall']} | {item['fdr']} | {item['tp']} | {item['fp']} | {item['fn']} |"
        )
    (output_dir / "official_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
