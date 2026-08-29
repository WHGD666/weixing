"""Official Docker entrypoint for submission version v1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from detector import Detector
from labels import CLASS_NAMES
from schema import validate_result_document

SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
DEFAULT_MODEL = Path("/app/models/best.pt")


def check_gpu() -> None:
    """Fail fast when the container cannot access a CUDA GPU."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; official evaluation requires GPU inference")
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)


def collect_images(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"input directory not found: {directory}")
    images = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise FileNotFoundError(f"no supported images found directly under {directory}")
    names = [path.name for path in images]
    if len(names) != len(set(names)):
        raise ValueError("input image file names must be unique")
    if len([path.stem for path in images]) != len(set(path.stem for path in images)):
        raise ValueError("input image stems must be unique")
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--tile-batch", type=int, default=4)
    return parser


def object_payload(item: dict[str, object]) -> dict[str, object]:
    category_id = int(item["category_id"])
    return {
        "category_id": category_id,
        "category_name": CLASS_NAMES[category_id],
        "score": round(float(item["score"]), 6),
        "bbox": [round(float(value), 4) for value in item["bbox"]],
    }


def main() -> None:
    args = build_parser().parse_args()
    check_gpu()
    image_paths = collect_images(args.input)
    output_dir = Path(args.output)
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
    )

    entries: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for image_path in image_paths:
        image_start = time.perf_counter()
        width, height, detections = detector.predict_image(image_path)
        run_end_timestamp = int(time.time() * 1000)
        image_seconds = time.perf_counter() - image_start
        entries.append(
            {
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "run_end_timestamp": run_end_timestamp,
                "objects": [object_payload(item) for item in sorted(detections, key=lambda item: float(item["score"]), reverse=True)],
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

    result = {"status": "success", "images": entries}
    validate_result_document(result, [path.name for path in image_paths])
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total_seconds = time.perf_counter() - total_start
    timing_result = {
        "image_count": len(timings),
        "total_seconds": round(total_seconds, 6),
        "max_image_seconds": round(max(item["seconds"] for item in timings), 6),
        "images": timings,
    }
    (output_dir / "timings.json").write_text(json.dumps(timing_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"images={len(entries)}", flush=True)
    print(f"total_seconds={total_seconds:.3f}", flush=True)
    print(f"max_image_seconds={timing_result['max_image_seconds']:.3f}", flush=True)
    print(f"result={output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
