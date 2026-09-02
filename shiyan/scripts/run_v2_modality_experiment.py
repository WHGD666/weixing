"""Run the frozen v2 detector with image-modality-aware class thresholds.

This is a diagnostic candidate, not a submission entrypoint. It keeps the v2
model and tiled detector unchanged, then applies per-image thresholds based on
whether the source image is effectively grayscale or colored.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIP_CLASS_IDS = frozenset(range(0, 4))
AIRCRAFT_CLASS_IDS = frozenset(range(4, 24))
VEHICLE_CLASS_IDS = frozenset({24})
FSC_CLASS_ID = 24
SUPPORTED_POLICIES = ("keep", "threshold", "drop")


def image_modality(
    image_path: Path,
    *,
    gray_mean_range_threshold: float,
    gray_pixel_range_threshold: float,
    gray_pixel_fraction_threshold: float,
) -> dict[str, Any]:
    """Measure channel separation and classify one image without using labels."""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {image_path}")
    channel_range = image.max(axis=2).astype(np.float32) - image.min(axis=2).astype(np.float32)
    mean_range = float(channel_range.mean() / 255.0)
    colored_fraction = float(
        np.mean(channel_range > gray_pixel_range_threshold * 255.0)
    )
    is_grayscale = (
        mean_range <= gray_mean_range_threshold
        and colored_fraction <= gray_pixel_fraction_threshold
    )
    return {
        "modality": "grayscale" if is_grayscale else "color",
        "is_grayscale": is_grayscale,
        "mean_channel_range": round(mean_range, 8),
        "colored_pixel_fraction": round(colored_fraction, 8),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def class_threshold(
    category_id: int,
    *,
    modality: str,
    ship_gray_conf: float,
    ship_color_conf: float,
    color_ship_policy: str,
    aircraft_conf: float,
    fsc_conf: float,
) -> float | None:
    """Return the post-inference threshold; None means drop the class."""

    if category_id in SHIP_CLASS_IDS:
        if modality == "grayscale":
            return ship_gray_conf
        if color_ship_policy == "drop":
            return None
        if color_ship_policy == "keep":
            return ship_gray_conf
        return ship_color_conf
    if category_id in AIRCRAFT_CLASS_IDS:
        return aircraft_conf
    if category_id in VEHICLE_CLASS_IDS:
        return fsc_conf
    raise ValueError(f"unknown category_id: {category_id}")


def filter_detections(
    detections: list[dict[str, Any]],
    *,
    modality: str,
    ship_gray_conf: float,
    ship_color_conf: float,
    color_ship_policy: str,
    aircraft_conf: float,
    fsc_conf: float,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for detection in detections:
        category_id = int(detection["category_id"])
        threshold = class_threshold(
            category_id,
            modality=modality,
            ship_gray_conf=ship_gray_conf,
            ship_color_conf=ship_color_conf,
            color_ship_policy=color_ship_policy,
            aircraft_conf=aircraft_conf,
            fsc_conf=fsc_conf,
        )
        if threshold is not None and float(detection["score"]) >= threshold:
            filtered.append(dict(detection))
    return filtered


def _object_payload(item: dict[str, Any], class_names: tuple[str, ...]) -> dict[str, Any]:
    category_id = int(item["category_id"])
    return {
        "category_id": category_id,
        "category_name": class_names[category_id],
        "score": round(float(item["score"]), 6),
        "bbox": [round(float(value), 4) for value in item["bbox"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-list", required=True, help="fixed validation image list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=0, help="0 means all images")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.10, help="v2 model pre-filter; use <= all active thresholds")
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--strategy", choices=("soft", "strict"), default="soft")
    parser.add_argument("--ship-gray-conf", type=float, default=0.20)
    parser.add_argument("--ship-color-conf", type=float, default=0.60)
    parser.add_argument("--color-ship-policy", choices=SUPPORTED_POLICIES, default="threshold")
    parser.add_argument("--aircraft-conf", type=float, default=0.30)
    parser.add_argument("--fsc-conf", type=float, default=0.35)
    parser.add_argument("--gray-mean-range-threshold", type=float, default=0.02)
    parser.add_argument("--gray-pixel-range-threshold", type=float, default=0.04)
    parser.add_argument("--gray-pixel-fraction-threshold", type=float, default=0.05)
    return parser


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "conf",
        "ship_gray_conf",
        "ship_color_conf",
        "aircraft_conf",
        "fsc_conf",
        "gray_mean_range_threshold",
        "gray_pixel_range_threshold",
        "gray_pixel_fraction_threshold",
    ):
        _validate_probability(name, float(getattr(args, name)))
    if args.strategy == "strict":
        args.color_ship_policy = "drop"

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    app_dir = REPO_ROOT / "submit" / "v2" / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    from detector import Detector  # noqa: PLC0415
    from labels import CLASS_NAMES  # noqa: PLC0415
    from shiyan.src.inference.runner import read_image_list  # noqa: PLC0415
    from shiyan.src.inference.schema import validate_result_document  # noqa: PLC0415

    all_image_paths = read_image_list(args.image_list)
    if args.sample_count < 0:
        raise ValueError("sample-count must be non-negative")
    if args.sample_count == 0 or args.sample_count >= len(all_image_paths):
        image_paths = all_image_paths
    elif args.sample_count == 1:
        image_paths = [all_image_paths[0]]
    else:
        indexes = [round(index * (len(all_image_paths) - 1) / (args.sample_count - 1)) for index in range(args.sample_count)]
        image_paths = [all_image_paths[index] for index in indexes]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = Detector(
        args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        mode=args.mode,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        merge_iou=args.merge_iou,
        tile_batch=args.tile_batch,
        class_thresholds={FSC_CLASS_ID: args.fsc_conf},
    )

    entries: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for image_path in image_paths:
        modality = image_modality(
            image_path,
            gray_mean_range_threshold=args.gray_mean_range_threshold,
            gray_pixel_range_threshold=args.gray_pixel_range_threshold,
            gray_pixel_fraction_threshold=args.gray_pixel_fraction_threshold,
        )
        image_start = time.perf_counter()
        raw_detections = detector.predict_image(image_path)[2]
        kept_detections = filter_detections(
            raw_detections,
            modality=modality["modality"],
            ship_gray_conf=args.ship_gray_conf,
            ship_color_conf=args.ship_color_conf,
            color_ship_policy=args.color_ship_policy,
            aircraft_conf=args.aircraft_conf,
            fsc_conf=args.fsc_conf,
        )
        image_seconds = time.perf_counter() - image_start
        width, height = modality["width"], modality["height"]
        entries.append(
            {
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "run_end_timestamp": int(time.time() * 1000),
                "objects": [
                    _object_payload(item, CLASS_NAMES)
                    for item in sorted(kept_detections, key=lambda item: float(item["score"]), reverse=True)
                ],
            }
        )
        timings.append(
            {
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "seconds": round(image_seconds, 6),
                "objects": len(kept_detections),
            }
        )
        modality_rows.append(
            {
                "file_name": image_path.name,
                "modality": modality["modality"],
                "mean_channel_range": modality["mean_channel_range"],
                "colored_pixel_fraction": modality["colored_pixel_fraction"],
                "raw_objects": len(raw_detections),
                "kept_objects": len(kept_detections),
            }
        )

    result = {"status": "success", "images": entries}
    validate_result_document(result, [path.name for path in image_paths])
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total_seconds = time.perf_counter() - total_start
    timing_result = {
        "image_count": len(timings),
        "total_seconds": round(total_seconds, 6),
        "max_image_seconds": round(max(item["seconds"] for item in timings), 6),
        "images": timings,
    }
    (output_dir / "timings.json").write_text(
        json.dumps(timing_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "image_list.txt").write_text(
        "\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8"
    )
    with (output_dir / "modality_by_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(modality_rows[0]))
        writer.writeheader()
        writer.writerows(modality_rows)
    counts = {"grayscale": 0, "color": 0}
    for row in modality_rows:
        counts[str(row["modality"])] += 1
    metadata = {
        "experiment_type": "diagnostic_modality_aware_postprocess",
        "detector_source": "submit/v2/app/detector.py",
        "model": str(Path(args.model)),
        "strategy": args.strategy,
        "color_ship_policy": args.color_ship_policy,
        "thresholds": {
            "ship_gray": args.ship_gray_conf,
            "ship_color": args.ship_color_conf,
            "aircraft": args.aircraft_conf,
            "fsc": args.fsc_conf,
        },
        "gray_rule": {
            "mean_channel_range_lte": args.gray_mean_range_threshold,
            "colored_pixel_range_gt": args.gray_pixel_range_threshold,
            "colored_pixel_fraction_lte": args.gray_pixel_fraction_threshold,
        },
        "image_count": len(image_paths),
        "modality_counts": counts,
        "total_seconds": round(total_seconds, 6),
    }
    (output_dir / "modality_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"images={len(entries)}")
    print(f"grayscale_images={counts['grayscale']}")
    print(f"color_images={counts['color']}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f"max_image_seconds={timing_result['max_image_seconds']:.3f}")
    print(f"result={output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
