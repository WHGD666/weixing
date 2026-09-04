"""Sweep group confidence thresholds against two frozen label protocols."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.scripts.evaluate_official import (  # noqa: E402
    _aggregate,
    _iou,
    _new_stats,
    _read_ground_truth,
    _threshold,
)
from shiyan.scripts.evaluate_two_label_protocols import read_rename_manifest  # noqa: E402
from shiyan.src.inference.labels import (  # noqa: E402
    AIRCRAFT_CLASS_IDS,
    CLASS_NAMES,
    SHIP_CLASS_IDS,
    VEHICLE_CLASS_IDS,
    class_group,
)
from shiyan.src.inference.runner import read_image_list  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402

GROUP_CLASS_IDS = {
    "ship": SHIP_CLASS_IDS,
    "aircraft": AIRCRAFT_CLASS_IDS,
    "vehicle": VEHICLE_CLASS_IDS,
}


def parse_values(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("threshold list must not be empty")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("thresholds must be in [0, 1]")
    return values


def parse_class_grids(raw_values: list[str]) -> dict[int, list[float]]:
    grids: dict[int, list[float]] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"class threshold grid must use ID=VALUES: {raw}")
        category_text, values_text = raw.split("=", 1)
        category_id = int(category_text)
        if not 0 <= category_id < len(CLASS_NAMES):
            raise ValueError(f"category ID outside 0-{len(CLASS_NAMES) - 1}: {category_id}")
        if category_id in grids:
            raise ValueError(f"duplicate class threshold grid: {category_id}")
        grids[category_id] = parse_values(values_text)
    return grids


def load_protocol_ground_truth(
    image_paths: list[Path],
    entries: dict[str, dict[str, Any]],
    *,
    canonical_names: list[str],
    canonical_to_protocol: dict[str, str] | None = None,
) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    paths_by_name = {path.name: path for path in image_paths}
    ground_truth: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for canonical_name in canonical_names:
        protocol_name = (
            canonical_to_protocol[canonical_name]
            if canonical_to_protocol is not None
            else canonical_name
        )
        if protocol_name not in paths_by_name:
            raise ValueError(f"mapped validation image is missing: {protocol_name}")
        entry = entries[canonical_name]
        ground_truth[canonical_name] = _read_ground_truth(
            paths_by_name[protocol_name],
            int(entry["width"]),
            int(entry["height"]),
        )
    return ground_truth


def evaluate_thresholds(
    entries: dict[str, dict[str, Any]],
    canonical_names: list[str],
    ground_truth: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    consensus_thresholds: dict[str, float],
    single_thresholds: dict[str, float],
    class_thresholds: dict[int, float],
) -> dict[str, Any]:
    class_stats = [_new_stats() for _ in CLASS_NAMES]
    kept_objects = 0
    for name in canonical_names:
        truths = ground_truth[name]
        matched = [False] * len(truths)
        predictions = sorted(
            (
                item
                for item in entries[name]["objects"]
                if float(item["score"])
                >= max(
                    (
                        consensus_thresholds
                        if int(item.get("_support", 1)) >= 2
                        else single_thresholds
                    )[class_group(int(item["category_id"]))],
                    class_thresholds.get(int(item["category_id"]), 0.0),
                )
            ),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        kept_objects += len(predictions)
        for prediction in predictions:
            category_id = int(prediction["category_id"])
            prediction_box = tuple(float(value) for value in prediction["bbox"])
            best_index: int | None = None
            best_iou = 0.0
            for index, (truth_category, truth_box) in enumerate(truths):
                if matched[index] or truth_category != category_id:
                    continue
                overlap = _iou(prediction_box, truth_box)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index is not None and best_iou >= _threshold(category_id):
                matched[best_index] = True
                class_stats[category_id]["tp"] += 1
            else:
                class_stats[category_id]["fp"] += 1
        for index, (category_id, _) in enumerate(truths):
            if not matched[index]:
                class_stats[category_id]["fn"] += 1

    groups = {
        group: _aggregate(class_stats[index] for index in sorted(category_ids))
        for group, category_ids in GROUP_CLASS_IDS.items()
    }
    recalls = [float(item["recall"]) for item in groups.values() if item["recall"] is not None]
    fdrs = [float(item["fdr"]) for item in groups.values() if item["fdr"] is not None]
    return {
        "group_mean_recall": sum(recalls) / len(recalls),
        "group_mean_fdr": sum(fdrs) / len(fdrs),
        "kept_objects": kept_objects,
        "groups": groups,
    }


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and float(other["worst_recall"]) >= float(candidate["worst_recall"])
            and float(other["worst_fdr"]) <= float(candidate["worst_fdr"])
            and (
                float(other["worst_recall"]) > float(candidate["worst_recall"])
                or float(other["worst_fdr"]) < float(candidate["worst_fdr"])
            )
            for other in rows
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: (-float(row["worst_recall"]), float(row["worst_fdr"])))


def flattened_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        key: row[key]
        for key in (
            "ship_threshold",
            "aircraft_threshold",
            "vehicle_threshold",
            "ship_single_threshold",
            "aircraft_single_threshold",
            "vehicle_single_threshold",
            "worst_recall",
            "worst_fdr",
            "recall_margin",
            "fdr_margin",
            "gate_margin",
            "passes_both",
            "class_thresholds",
        )
    }
    for protocol in ("d0_original", "d3_manual"):
        metrics = row[protocol]
        flat[f"{protocol}_recall"] = metrics["group_mean_recall"]
        flat[f"{protocol}_fdr"] = metrics["group_mean_fdr"]
        flat[f"{protocol}_kept_objects"] = metrics["kept_objects"]
        for group in GROUP_CLASS_IDS:
            flat[f"{protocol}_{group}_recall"] = metrics["groups"][group]["recall"]
            flat[f"{protocol}_{group}_fdr"] = metrics["groups"][group]["fdr"]
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--timings")
    parser.add_argument("--original-image-list", required=True)
    parser.add_argument("--revised-image-list", required=True)
    parser.add_argument("--rename-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ship-thresholds", required=True)
    parser.add_argument("--aircraft-thresholds", required=True)
    parser.add_argument("--vehicle-thresholds", required=True)
    parser.add_argument(
        "--ship-single-thresholds",
        help="single-model thresholds; defaults to --ship-thresholds",
    )
    parser.add_argument(
        "--aircraft-single-thresholds",
        help="single-model thresholds; defaults to --aircraft-thresholds",
    )
    parser.add_argument(
        "--vehicle-single-thresholds",
        help="single-model thresholds; defaults to --vehicle-thresholds",
    )
    parser.add_argument(
        "--class-threshold-grid",
        action="append",
        default=[],
        help="optional fine-class override grid as ID=VALUE,VALUE; repeat per class",
    )
    parser.add_argument("--minimum-recall", type=float, default=0.85)
    parser.add_argument("--maximum-fdr", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    document = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    original_paths = read_image_list(args.original_image_list)
    canonical_names = [path.name for path in original_paths]
    validate_result_document(document, canonical_names)
    entries = {str(entry["file_name"]): entry for entry in document["images"]}

    mapping = read_rename_manifest(Path(args.rename_manifest))
    revised_paths = read_image_list(args.revised_image_list)
    d0_ground_truth = load_protocol_ground_truth(
        original_paths,
        entries,
        canonical_names=canonical_names,
    )
    d3_ground_truth = load_protocol_ground_truth(
        revised_paths,
        entries,
        canonical_names=canonical_names,
        canonical_to_protocol=mapping,
    )

    ship_values = parse_values(args.ship_thresholds)
    aircraft_values = parse_values(args.aircraft_thresholds)
    vehicle_values = parse_values(args.vehicle_thresholds)
    ship_single_values = parse_values(args.ship_single_thresholds or args.ship_thresholds)
    aircraft_single_values = parse_values(
        args.aircraft_single_thresholds or args.aircraft_thresholds
    )
    vehicle_single_values = parse_values(args.vehicle_single_thresholds or args.vehicle_thresholds)
    class_grids = parse_class_grids(args.class_threshold_grid)
    class_ids = sorted(class_grids)
    class_combinations = list(
        itertools.product(*(class_grids[category_id] for category_id in class_ids))
    ) or [()]
    rows: list[dict[str, Any]] = []
    for (
        ship_threshold,
        aircraft_threshold,
        vehicle_threshold,
        ship_single_threshold,
        aircraft_single_threshold,
        vehicle_single_threshold,
        class_values,
    ) in itertools.product(
        ship_values,
        aircraft_values,
        vehicle_values,
        ship_single_values,
        aircraft_single_values,
        vehicle_single_values,
        class_combinations,
    ):
        consensus_thresholds = {
            "ship": ship_threshold,
            "aircraft": aircraft_threshold,
            "vehicle": vehicle_threshold,
        }
        single_thresholds = {
            "ship": ship_single_threshold,
            "aircraft": aircraft_single_threshold,
            "vehicle": vehicle_single_threshold,
        }
        class_thresholds = dict(zip(class_ids, class_values))
        d0 = evaluate_thresholds(
            entries,
            canonical_names,
            d0_ground_truth,
            consensus_thresholds,
            single_thresholds,
            class_thresholds,
        )
        d3 = evaluate_thresholds(
            entries,
            canonical_names,
            d3_ground_truth,
            consensus_thresholds,
            single_thresholds,
            class_thresholds,
        )
        worst_recall = min(float(d0["group_mean_recall"]), float(d3["group_mean_recall"]))
        worst_fdr = max(float(d0["group_mean_fdr"]), float(d3["group_mean_fdr"]))
        recall_margin = worst_recall - args.minimum_recall
        fdr_margin = args.maximum_fdr - worst_fdr
        rows.append(
            {
                "ship_threshold": ship_threshold,
                "aircraft_threshold": aircraft_threshold,
                "vehicle_threshold": vehicle_threshold,
                "ship_single_threshold": ship_single_threshold,
                "aircraft_single_threshold": aircraft_single_threshold,
                "vehicle_single_threshold": vehicle_single_threshold,
                "worst_recall": worst_recall,
                "worst_fdr": worst_fdr,
                "recall_margin": recall_margin,
                "fdr_margin": fdr_margin,
                "gate_margin": min(recall_margin, fdr_margin),
                "passes_both": recall_margin >= 0.0 and fdr_margin >= 0.0,
                "class_thresholds": json.dumps(class_thresholds, sort_keys=True),
                "d0_original": d0,
                "d3_manual": d3,
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["passes_both"]),
            -float(row["gate_margin"]),
            -float(row["worst_recall"]),
            float(row["worst_fdr"]),
        ),
    )
    pareto = pareto_front(rows)
    timing = None
    if args.timings:
        timing_document = json.loads(Path(args.timings).read_text(encoding="utf-8"))
        timing = {
            "max_image_seconds": float(timing_document["max_image_seconds"]),
            "total_seconds": float(timing_document["total_seconds"]),
        }
    output = {
        "evaluation_scope": "internal_validation_threshold_sweep",
        "official_hidden_test": False,
        "source_predictions": str(Path(args.predictions)),
        "grid": {
            "ship": ship_values,
            "aircraft": aircraft_values,
            "vehicle": vehicle_values,
            "single_ship": ship_single_values,
            "single_aircraft": aircraft_single_values,
            "single_vehicle": vehicle_single_values,
            "class_overrides": class_grids,
            "candidate_count": len(rows),
        },
        "gates": {"minimum_recall": args.minimum_recall, "maximum_fdr": args.maximum_fdr},
        "timing": timing,
        "passing_count": sum(bool(row["passes_both"]) for row in rows),
        "ranked": ranked,
        "pareto": pareto,
    }
    (output_dir / "threshold_sweep.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_rows = [flattened_row(row) for row in ranked]
    with (output_dir / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    lines = [
        "# Dual-Protocol Group-Threshold Sweep",
        "",
        "Internal validation only; this is not an official hidden-test result.",
        "",
        f"- Candidates: {len(rows)}",
        f"- Passing both label protocols: {output['passing_count']}",
        f"- Pareto candidates: {len(pareto)}",
    ]
    if timing is not None:
        lines.append(f"- Cached maximum image time: {timing['max_image_seconds']:.6f} s")
    lines.extend(
        [
            "",
            "| Consensus S/A/V | Single S/A/V | Class overrides | Worst Recall | Worst FDR | Gate margin | Pass |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in ranked[: args.top_k]:
        lines.append(
            f"| {row['ship_threshold']:.3f}/{row['aircraft_threshold']:.3f}/"
            f"{row['vehicle_threshold']:.3f} | {row['ship_single_threshold']:.3f}/"
            f"{row['aircraft_single_threshold']:.3f}/{row['vehicle_single_threshold']:.3f} | "
            f"`{row['class_thresholds']}` | "
            f"{row['worst_recall']:.6f} | "
            f"{row['worst_fdr']:.6f} | {row['gate_margin']:.6f} | {row['passes_both']} |"
        )
    lines.extend(
        [
            "",
            "The ranking maximizes the smaller of the Recall and FDR gate margins. "
            "Use the Pareto set and a separate audit run before freezing a submission candidate.",
        ]
    )
    (output_dir / "threshold_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    best = ranked[0]
    print(
        json.dumps(
            {
                "candidates": len(rows),
                "passing": output["passing_count"],
                "best": {
                    "ship": best["ship_threshold"],
                    "aircraft": best["aircraft_threshold"],
                    "vehicle": best["vehicle_threshold"],
                    "single_ship": best["ship_single_threshold"],
                    "single_aircraft": best["aircraft_single_threshold"],
                    "single_vehicle": best["vehicle_single_threshold"],
                    "class_thresholds": best["class_thresholds"],
                    "worst_recall": best["worst_recall"],
                    "worst_fdr": best["worst_fdr"],
                    "gate_margin": best["gate_margin"],
                },
            },
            ensure_ascii=False,
        )
    )
    print(f"report={output_dir / 'threshold_sweep.md'}")


if __name__ == "__main__":
    main()
