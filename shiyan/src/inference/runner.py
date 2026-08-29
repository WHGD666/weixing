"""Filesystem runner and official result.json writer."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .labels import CLASS_NAMES
from .predictor import Predictor
from .schema import validate_result_document
from .types import Detection

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass(frozen=True)
class RunSummary:
    image_count: int
    total_seconds: float
    max_image_seconds: float
    result_path: Path
    timing_path: Path


def collect_input_images(input_dir: str | Path) -> list[Path]:
    """Collect only first-level supported images, matching the official contract."""

    directory = Path(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"input directory not found: {directory}")
    paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not paths:
        raise FileNotFoundError(f"no supported images found directly under {directory}")
    return paths


def read_image_list(image_list: str | Path) -> list[Path]:
    """Read a newline-separated image list, resolving relative paths from cwd."""

    path = Path(image_list)
    if not path.is_file():
        raise FileNotFoundError(f"image list not found: {path}")
    result: list[Path] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        image_path = Path(value).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"image list line {line_number} not found: {image_path}")
        result.append(image_path)
    if not result:
        raise ValueError(f"image list is empty: {path}")
    return result


def evenly_spaced_sample(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    if count == 1:
        return [paths[0]]
    if count >= len(paths):
        return paths
    indexes = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[index] for index in indexes]


def _object_payload(detection: Detection) -> dict[str, object]:
    return {
        "category_id": detection.category_id,
        "category_name": CLASS_NAMES[detection.category_id],
        "score": round(float(detection.score), 6),
        "bbox": [round(float(value), 4) for value in detection.bbox],
    }


def _save_visualization(image_path: Path, detections: list[Detection], output_dir: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image for visualization: {image_path}")
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        label = f"{CLASS_NAMES[detection.category_id]} {detection.score:.3f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_dir / image_path.name), image):
        raise OSError(f"unable to write visualization: {image_path.name}")


def run_predictions(
    image_paths: list[Path],
    predictor: Predictor,
    output_dir: str | Path,
    *,
    save_visualizations: bool = False,
) -> RunSummary:
    """Run the detector and write official result.json plus a local timing sidecar."""

    if not image_paths:
        raise ValueError("image_paths must not be empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = [path.name for path in image_paths]
    if len(names) != len(set(names)):
        raise ValueError("input image file names must be unique for official result.json")
    image_ids = [path.stem for path in image_paths]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("input image stems must be unique for official result.json")

    entries: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for image_path in image_paths:
        image_start = time.perf_counter()
        width, height, detections = predictor.predict_image(image_path)
        image_seconds = time.perf_counter() - image_start
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
                "seconds": round(image_seconds, 6),
                "objects": len(detections),
            }
        )
        if save_visualizations:
            _save_visualization(image_path, detections, output / "visualizations")

    total_seconds = time.perf_counter() - total_start
    document = {"status": "success", "images": entries}
    validate_result_document(document, names)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timing_path = output / "timings.json"
    timing_document = {
        "image_count": len(timings),
        "total_seconds": round(total_seconds, 6),
        "max_image_seconds": round(max(item["seconds"] for item in timings), 6),
        "images": timings,
    }
    timing_path.write_text(json.dumps(timing_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return RunSummary(
        image_count=len(entries),
        total_seconds=total_seconds,
        max_image_seconds=max(item["seconds"] for item in timings),
        result_path=result_path,
        timing_path=timing_path,
    )
