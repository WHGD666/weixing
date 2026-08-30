"""Analyze confidence bands and near-duplicate predictions on a fixed result."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.scripts.analyze_errors import _match_image  # noqa: E402
from shiyan.scripts.evaluate_official import _iou  # noqa: E402
from shiyan.src.inference.labels import CLASS_NAMES, class_group  # noqa: E402
from shiyan.src.inference.runner import read_image_list  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


DEFAULT_BIN_EDGES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01)


def _bin_label(lower: float, upper: float) -> str:
    return f"[{lower:.2f},{upper:.2f})"


def _score_bin(score: float, edges: tuple[float, ...]) -> tuple[float, float] | None:
    for lower, upper in zip(edges, edges[1:]):
        if lower <= score < upper:
            return lower, upper
    return None


def _parse_edges(value: str) -> tuple[float, ...]:
    edges = tuple(float(item) for item in value.split(","))
    if len(edges) < 2 or any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("--bin-edges must be strictly increasing and contain at least two values")
    if edges[0] < 0.0 or edges[-1] <= 1.0:
        raise ValueError("--bin-edges must start at >= 0 and end above 1.0")
    return edges


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bin-edges",
        default=",".join(str(edge) for edge in DEFAULT_BIN_EDGES),
        help="comma-separated score-bin edges; the final edge must be above 1.0",
    )
    parser.add_argument("--duplicate-iou", type=float, default=0.30)
    args = parser.parse_args()
    if not 0.0 < args.duplicate_iou <= 1.0:
        raise ValueError("duplicate-iou must be in (0, 1]")

    edges = _parse_edges(args.bin_edges)
    image_paths = read_image_list(args.image_list)
    document = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    validate_result_document(document, [path.name for path in image_paths])
    entries = {entry["file_name"]: entry for entry in document["images"]}

    band_stats: dict[tuple[int, float, float], dict[str, int]] = defaultdict(
        lambda: {"predictions": 0, "tp": 0, "fp": 0}
    )
    duplicate_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"images_with_pairs": 0, "pair_count": 0, "tp_fp_pairs": 0, "fp_fp_pairs": 0, "tp_tp_pairs": 0}
    )
    total_predictions = 0
    total_duplicate_pairs = 0

    for image_path in image_paths:
        match = _match_image(image_path, entries[image_path.name])
        predictions = match["predictions"]
        matched_prediction = match["matched_prediction"]
        total_predictions += len(predictions)
        image_pair_categories: set[int] = set()
        for prediction, matched in zip(predictions, matched_prediction):
            category_id = int(prediction["category_id"])
            score = float(prediction["score"])
            score_bin = _score_bin(score, edges)
            if score_bin is None:
                raise ValueError(f"score {score} is outside configured bins for {image_path.name}")
            stats = band_stats[(category_id, *score_bin)]
            stats["predictions"] += 1
            stats["tp" if matched else "fp"] += 1

        for left_index, left in enumerate(predictions):
            left_category = int(left["category_id"])
            left_box = tuple(float(value) for value in left["bbox"])
            for right_index in range(left_index + 1, len(predictions)):
                right = predictions[right_index]
                if int(right["category_id"]) != left_category:
                    continue
                right_box = tuple(float(value) for value in right["bbox"])
                if _iou(left_box, right_box) < args.duplicate_iou:
                    continue
                left_matched = bool(matched_prediction[left_index])
                right_matched = bool(matched_prediction[right_index])
                pair_kind = "tp_tp_pairs" if left_matched and right_matched else "tp_fp_pairs" if left_matched or right_matched else "fp_fp_pairs"
                duplicate_stats[left_category]["pair_count"] += 1
                duplicate_stats[left_category][pair_kind] += 1
                image_pair_categories.add(left_category)
                total_duplicate_pairs += 1
        for category_id in image_pair_categories:
            duplicate_stats[category_id]["images_with_pairs"] += 1

    band_rows: list[dict[str, object]] = []
    for category_id in range(len(CLASS_NAMES)):
        for lower, upper in zip(edges, edges[1:]):
            stats = band_stats[(category_id, lower, upper)]
            predictions = stats["predictions"]
            tp, fp = stats["tp"], stats["fp"]
            band_rows.append(
                {
                    "category_id": category_id,
                    "category_name": CLASS_NAMES[category_id],
                    "group": class_group(category_id),
                    "score_bin": _bin_label(lower, upper),
                    "score_lower": lower,
                    "score_upper": upper,
                    "predictions": predictions,
                    "tp": tp,
                    "fp": fp,
                    "precision": tp / predictions if predictions else None,
                    "fdr": fp / predictions if predictions else None,
                }
            )

    duplicate_rows = [
        {
            "category_id": category_id,
            "category_name": CLASS_NAMES[category_id],
            "group": class_group(category_id),
            **stats,
        }
        for category_id, stats in sorted(duplicate_stats.items())
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "confidence_bins.csv",
        [
            "category_id",
            "category_name",
            "group",
            "score_bin",
            "score_lower",
            "score_upper",
            "predictions",
            "tp",
            "fp",
            "precision",
            "fdr",
        ],
        band_rows,
    )
    _write_csv(
        output_dir / "near_duplicate_summary.csv",
        ["category_id", "category_name", "group", "images_with_pairs", "pair_count", "tp_fp_pairs", "fp_fp_pairs", "tp_tp_pairs"],
        duplicate_rows,
    )
    summary = {
        "evaluation_scope": "internal_fixed_prediction_diagnostic",
        "image_count": len(image_paths),
        "total_predictions": total_predictions,
        "duplicate_iou": args.duplicate_iou,
        "total_duplicate_pairs": total_duplicate_pairs,
        "bin_edges": edges,
        "source_predictions": str(Path(args.predictions)),
        "source_image_list": str(Path(args.image_list)),
        "outputs": ["confidence_bins.csv", "near_duplicate_summary.csv"],
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"images={len(image_paths)}")
    print(f"predictions={total_predictions}")
    print(f"near_duplicate_pairs={total_duplicate_pairs}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
