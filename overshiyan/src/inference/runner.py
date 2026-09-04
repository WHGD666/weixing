from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .labels import CLASS_NAMES
from .predictor import Predictor
from .schema import validate_result_document
from .types import Detection

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class RunSummary:
    image_count: int
    total_seconds: float
    max_image_seconds: float
    result_path: Path
    timing_path: Path


def collect_input_images(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"input directory not found: {directory}")
    paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not paths:
        raise FileNotFoundError(f"no supported images directly under {directory}")
    return paths


def _object_payload(detection: Detection) -> dict[str, object]:
    return {
        "category_id": detection.category_id,
        "category_name": CLASS_NAMES[detection.category_id],
        "score": round(float(detection.score), 6),
        "bbox": [round(float(value), 4) for value in detection.bbox],
    }


def run_predictions(
    image_paths: list[Path], predictor: Predictor, output_dir: str | Path
) -> RunSummary:
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    names = [path.name for path in image_paths]
    stems = [path.stem for path in image_paths]
    if len(names) != len(set(names)) or len(stems) != len(set(stems)):
        raise ValueError("input image names and stems must be unique")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        image_start = time.perf_counter()
        width, height, detections = predictor.predict_image(image_path)
        elapsed = time.perf_counter() - image_start
        entries.append(
            {
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "run_end_timestamp": int(time.time() * 1000),
                "objects": [_object_payload(item) for item in sorted(detections, key=lambda item: item.score, reverse=True)],
            }
        )
        timings.append(
            {
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "seconds": round(elapsed, 6),
                "objects": len(detections),
            }
        )
        if index % 100 == 0 or index == len(image_paths):
            print(f"completed={index}/{len(image_paths)}", flush=True)
    total_seconds = time.perf_counter() - total_start
    document = {"status": "success", "images": entries}
    validate_result_document(document, names)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timing_path = output / "timings.json"
    timing_document = {
        "image_count": len(timings),
        "total_seconds": round(total_seconds, 6),
        "average_image_seconds": round(total_seconds / len(timings), 6),
        "max_image_seconds": round(max(item["seconds"] for item in timings), 6),
        "images": timings,
    }
    timing_path.write_text(json.dumps(timing_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return RunSummary(
        image_count=len(entries),
        total_seconds=total_seconds,
        max_image_seconds=float(timing_document["max_image_seconds"]),
        result_path=result_path,
        timing_path=timing_path,
    )
