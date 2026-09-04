"""Generate official-format candidates from a named multi-model prediction cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.src.inference.algorithmic_postprocess import (  # noqa: E402
    apply_modality_and_thresholds,
    combine_multiple_models,
    group_aware_nms,
    official_object,
)
from shiyan.src.inference.labels import CLASS_NAMES  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


def parse_model_subset(raw: str, available: list[str]) -> list[str]:
    selected = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not selected:
        return available
    if len(selected) != len(set(selected)):
        raise ValueError("model subset contains duplicates")
    missing = [item for item in selected if item not in available]
    if missing:
        raise ValueError(f"models are not present in cache: {missing}")
    return selected


def parse_class_threshold(values: list[str]) -> dict[int, float]:
    parsed: dict[int, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"class threshold must use ID=VALUE: {raw}")
        category_text, threshold_text = raw.split("=", 1)
        category_id = int(category_text)
        threshold = float(threshold_text)
        if not 0 <= category_id < len(CLASS_NAMES):
            raise ValueError(f"category ID outside 0-{len(CLASS_NAMES) - 1}: {category_id}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]: {raw}")
        parsed[category_id] = threshold
    return parsed


def group_values(args: argparse.Namespace, suffix: str) -> dict[str, float]:
    values = {
        "ship": float(getattr(args, f"ship_{suffix}")),
        "aircraft": float(getattr(args, f"aircraft_{suffix}")),
        "vehicle": float(getattr(args, f"vehicle_{suffix}")),
    }
    for group, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{group}-{suffix.replace('_', '-')} must be in [0, 1]")
    return values


def used_sources(
    fusion_mode: str,
    selected_models: list[str],
    source_preference: dict[str, str],
) -> list[str]:
    if fusion_mode == "route":
        return list(dict.fromkeys(source_preference.values()))
    return selected_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="", help="comma-separated cache model names; empty means all")
    parser.add_argument(
        "--fusion-mode",
        choices=("route", "union", "consensus", "vote"),
        default="consensus",
    )
    parser.add_argument("--consensus-iou", type=float, default=0.50)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--score-aggregation", choices=("max", "mean"), default="max")
    parser.add_argument("--ship-source", default="a")
    parser.add_argument("--aircraft-source", default="a")
    parser.add_argument("--vehicle-source", default="b")
    parser.add_argument("--modality-policy", choices=("off", "soft", "strict"), default="off")
    parser.add_argument("--ship-consensus-conf", type=float, default=0.20)
    parser.add_argument("--aircraft-consensus-conf", type=float, default=0.20)
    parser.add_argument("--vehicle-consensus-conf", type=float, default=0.25)
    parser.add_argument("--ship-single-conf", type=float, default=0.45)
    parser.add_argument("--aircraft-single-conf", type=float, default=0.35)
    parser.add_argument("--vehicle-single-conf", type=float, default=0.45)
    parser.add_argument("--ship-color-conf", type=float, default=0.60)
    parser.add_argument("--nonship-gray-conf", type=float, default=0.60)
    parser.add_argument("--class-threshold", action="append", default=[])
    parser.add_argument("--group-nms-iou", type=float, default=0.50)
    parser.add_argument("--disable-group-nms", action="store_true")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--retain-debug-fields",
        action="store_true",
        help="retain internal support/source fields for offline calibration only",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.is_file():
        raise FileNotFoundError(f"cache file not found: {cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("schema_version") != "multi_model_cache_v1" or cache.get("status") != "success":
        raise ValueError("cache does not satisfy multi_model_cache_v1")
    if not isinstance(cache.get("images"), list) or not cache["images"]:
        raise ValueError("cache contains no images")
    available_models = [str(item) for item in cache["metadata"]["model_order"]]
    selected_models = parse_model_subset(args.models, available_models)
    if not 0.0 < args.consensus_iou <= 1.0:
        raise ValueError("consensus-iou must be in (0, 1]")
    if not 0.0 < args.group_nms_iou <= 1.0:
        raise ValueError("group-nms-iou must be in (0, 1]")
    if not 1 <= args.minimum_support <= len(selected_models):
        raise ValueError("minimum-support must be between 1 and the selected model count")
    if args.max_det <= 0:
        raise ValueError("max-det must be positive")
    for name in ("ship_color_conf", "nonship_gray_conf"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; use a new run ID: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_preference = {
        "ship": args.ship_source.lower(),
        "aircraft": args.aircraft_source.lower(),
        "vehicle": args.vehicle_source.lower(),
    }
    for group, source in source_preference.items():
        if source != "max" and source not in selected_models:
            raise ValueError(f"{group} source is not selected: {source}")
    if args.fusion_mode == "route" and "max" in source_preference.values():
        raise ValueError("route mode requires an explicit model for every group")
    consensus_thresholds = group_values(args, "consensus_conf")
    single_thresholds = group_values(args, "single_conf")
    class_thresholds = parse_class_threshold(args.class_threshold)
    timing_sources = used_sources(args.fusion_mode, selected_models, source_preference)

    output_images: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    raw_objects = 0
    combined_objects = 0
    kept_objects = 0
    support_counts: dict[int, int] = {}
    modality_counts = {"grayscale": 0, "color": 0, "uncertain": 0}
    postprocess_start = time.perf_counter()
    for entry in cache["images"]:
        modality = str(entry["modality"]["modality"])
        modality_counts[modality] += 1
        model_outputs = {source: entry["models"][source] for source in selected_models}
        raw_objects += sum(len(items) for items in model_outputs.values())
        combined = combine_multiple_models(
            model_outputs,
            fusion_mode=args.fusion_mode,
            consensus_iou=args.consensus_iou,
            source_preference=source_preference,
            minimum_support=args.minimum_support,
            score_aggregation=args.score_aggregation,
        )
        combined_objects += len(combined)
        for item in combined:
            support = int(item.get("support", 1))
            support_counts[support] = support_counts.get(support, 0) + 1
        filtered = apply_modality_and_thresholds(
            combined,
            modality=modality,
            modality_policy=args.modality_policy,
            consensus_thresholds=consensus_thresholds,
            single_thresholds=single_thresholds,
            ship_color_conf=args.ship_color_conf,
            nonship_gray_conf=args.nonship_gray_conf,
            class_thresholds=class_thresholds,
        )
        if args.disable_group_nms:
            filtered = sorted(filtered, key=lambda item: float(item["score"]), reverse=True)[: args.max_det]
        else:
            filtered = group_aware_nms(
                filtered,
                iou_threshold=args.group_nms_iou,
                max_detections=args.max_det,
            )
        kept_objects += len(filtered)
        output_images.append(
            {
                "image_id": entry["image_id"],
                "file_name": entry["file_name"],
                "width": entry["width"],
                "height": entry["height"],
                "run_end_timestamp": int(time.time() * 1000),
                "objects": [
                    {
                        **official_object(item, CLASS_NAMES),
                        **(
                            {
                                "_support": int(item.get("support", 1)),
                                "_sources": list(item.get("sources", ())),
                            }
                            if args.retain_debug_fields
                            else {}
                        ),
                    }
                    for item in sorted(filtered, key=lambda item: float(item["score"]), reverse=True)
                ],
            }
        )
        seconds = sum(float(entry["seconds"][source]) for source in timing_sources)
        timing_rows.append(
            {
                "file_name": entry["file_name"],
                "width": entry["width"],
                "height": entry["height"],
                "seconds": round(seconds, 6),
                "objects": len(filtered),
            }
        )

    postprocess_seconds = time.perf_counter() - postprocess_start
    result = {"status": "success", "images": output_images}
    names = [entry["file_name"] for entry in cache["images"]]
    validate_result_document(result, names)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    total_seconds = sum(float(row["seconds"]) for row in timing_rows) + postprocess_seconds
    timing_document = {
        "image_count": len(timing_rows),
        "total_seconds": round(total_seconds, 6),
        "max_image_seconds": round(max(float(row["seconds"]) for row in timing_rows), 6),
        "postprocess_total_seconds": round(postprocess_seconds, 6),
        "scope": "cached_inference_plus_offline_postprocess_proxy",
        "models_counted": timing_sources,
        "images": timing_rows,
    }
    (output_dir / "timings.json").write_text(
        json.dumps(timing_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    source_image_list = cache_path.parent / "image_list.txt"
    if not source_image_list.is_file():
        raise FileNotFoundError(f"cache image list is missing: {source_image_list}")
    (output_dir / "image_list.txt").write_text(
        source_image_list.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    configuration = {
        "experiment_type": "algorithmic_multi_model_postprocess",
        "cache": str(cache_path),
        "available_models": available_models,
        "selected_models": selected_models,
        "fusion_mode": args.fusion_mode,
        "consensus_iou": args.consensus_iou,
        "minimum_support": args.minimum_support,
        "score_aggregation": args.score_aggregation,
        "source_preference": source_preference,
        "modality_policy": args.modality_policy,
        "consensus_thresholds": consensus_thresholds,
        "single_thresholds": single_thresholds,
        "ship_color_conf": args.ship_color_conf,
        "nonship_gray_conf": args.nonship_gray_conf,
        "class_thresholds": class_thresholds,
        "retain_debug_fields": args.retain_debug_fields,
        "group_nms": {"enabled": not args.disable_group_nms, "iou": args.group_nms_iou},
        "counts": {
            "images": len(output_images),
            "raw_objects": raw_objects,
            "combined_objects": combined_objects,
            "kept_objects": kept_objects,
            "support": support_counts,
            "modality": modality_counts,
        },
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"images={len(output_images)}")
    print(f"selected_models={selected_models}")
    print(f"timing_models={timing_sources}")
    print(f"raw_objects={raw_objects}")
    print(f"combined_objects={combined_objects}")
    print(f"kept_objects={kept_objects}")
    print(f"support_counts={support_counts}")
    print(f"postprocess_seconds={postprocess_seconds:.3f}")
    print(f"result={output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
