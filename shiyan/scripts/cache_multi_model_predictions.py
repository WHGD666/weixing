"""Cache low-threshold predictions from named frozen models for offline experiments."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.src.inference.labels import CLASS_NAMES  # noqa: E402
from shiyan.src.inference.predictor import Predictor  # noqa: E402
from shiyan.src.inference.runner import evenly_spaced_sample, read_image_list  # noqa: E402
from shiyan.src.inference.types import Detection  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must use NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"invalid model name: {name}")
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        models[name] = path
    if len(models) < 2:
        raise ValueError("at least two named models are required")
    if len(set(models.values())) != len(models):
        raise ValueError("named models must point to different files")
    return models


def modality_stats(image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {image_path}")
    channel_range = image.max(axis=2).astype(np.float32) - image.min(axis=2).astype(np.float32)
    mean_range = float(channel_range.mean() / 255.0)
    colored_fraction = float(np.mean(channel_range > args.gray_pixel_range_threshold * 255.0))
    if mean_range <= args.gray_mean_range_threshold and colored_fraction <= args.gray_pixel_fraction_threshold:
        modality = "grayscale"
    elif mean_range >= args.color_mean_range_threshold or colored_fraction >= args.color_pixel_fraction_threshold:
        modality = "color"
    else:
        modality = "uncertain"
    return {
        "modality": modality,
        "mean_channel_range": round(mean_range, 8),
        "colored_pixel_fraction": round(colored_fraction, 8),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def object_payload(item: Detection) -> dict[str, Any]:
    return {
        "category_id": item.category_id,
        "category_name": CLASS_NAMES[item.category_id],
        "score": round(float(item.score), 8),
        "bbox": [round(float(value), 4) for value in item.bbox],
    }


def build_predictor(model: Path, args: argparse.Namespace) -> Predictor:
    return Predictor(
        model,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="repeat NAME=PATH")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=0, help="0 means all images")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--gray-mean-range-threshold", type=float, default=0.02)
    parser.add_argument("--gray-pixel-range-threshold", type=float, default=0.04)
    parser.add_argument("--gray-pixel-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--color-mean-range-threshold", type=float, default=0.04)
    parser.add_argument("--color-pixel-fraction-threshold", type=float, default=0.10)
    args = parser.parse_args()

    models = parse_models(args.model)
    if args.sample_count < 0:
        raise ValueError("sample-count must be non-negative")
    for name in (
        "conf",
        "iou",
        "tile_overlap",
        "merge_iou",
        "gray_mean_range_threshold",
        "gray_pixel_range_threshold",
        "gray_pixel_fraction_threshold",
        "color_mean_range_threshold",
        "color_pixel_fraction_threshold",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")
    if args.imgsz <= 0 or args.max_det <= 0 or args.tile_size <= 0 or args.tile_batch <= 0:
        raise ValueError("imgsz, max-det, tile-size and tile-batch must be positive")

    all_paths = read_image_list(args.image_list)
    image_paths = all_paths if args.sample_count == 0 else evenly_spaced_sample(all_paths, args.sample_count)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty; use a new cache ID: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    modality_counts = {"grayscale": 0, "color": 0, "uncertain": 0}
    entries: list[dict[str, Any]] = []
    for image_path in image_paths:
        modality = modality_stats(image_path, args)
        modality_counts[str(modality["modality"])] += 1
        entries.append(
            {
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "width": modality["width"],
                "height": modality["height"],
                "modality": modality,
                "models": {},
                "seconds": {},
            }
        )

    total_start = time.perf_counter()
    model_metadata: dict[str, dict[str, Any]] = {}
    for model_name, model_path in models.items():
        print(f"model_start={model_name} path={model_path}", flush=True)
        predictor = build_predictor(model_path, args)
        model_start = time.perf_counter()
        for index, (image_path, entry) in enumerate(zip(image_paths, entries), start=1):
            image_start = time.perf_counter()
            width, height, detections = predictor.predict_image(image_path)
            seconds = time.perf_counter() - image_start
            if (width, height) != (entry["width"], entry["height"]):
                raise ValueError(f"image dimensions disagree for {image_path.name}")
            entry["models"][model_name] = [object_payload(item) for item in detections]
            entry["seconds"][model_name] = round(seconds, 6)
            if index % 100 == 0 or index == len(entries):
                print(f"model={model_name} completed={index}/{len(entries)}", flush=True)
        model_seconds = time.perf_counter() - model_start
        model_metadata[model_name] = {
            "path": str(model_path),
            "sha256": file_sha256(model_path),
            "total_seconds": round(model_seconds, 6),
        }
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_seconds = time.perf_counter() - total_start
    timing_rows = [
        {
            "file_name": entry["file_name"],
            "seconds": entry["seconds"],
            "seconds_all_models": round(sum(float(value) for value in entry["seconds"].values()), 6),
        }
        for entry in entries
    ]
    cache = {
        "schema_version": "multi_model_cache_v1",
        "status": "success",
        "metadata": {
            "models": model_metadata,
            "model_order": list(models),
            "image_list": str(Path(args.image_list)),
            "image_count": len(entries),
            "modality_counts": modality_counts,
            "inference": {
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "mode": args.mode,
                "tile_size": args.tile_size,
                "tile_overlap": args.tile_overlap,
                "merge_iou": args.merge_iou,
                "tile_batch": args.tile_batch,
            },
        },
        "images": entries,
    }
    cache_path = output_dir / "raw_cache.json"
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "image_list.txt").write_text(
        "\n".join(str(path) for path in image_paths) + "\n",
        encoding="utf-8",
    )
    timing_document = {
        "image_count": len(timing_rows),
        "model_count": len(models),
        "total_wall_seconds": round(total_seconds, 6),
        "max_all_models_image_seconds": round(
            max(float(row["seconds_all_models"]) for row in timing_rows),
            6,
        ),
        "models": model_metadata,
        "images": timing_rows,
    }
    (output_dir / "cache_timings.json").write_text(
        json.dumps(timing_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"images={len(entries)}")
    print(f"models={list(models)}")
    print(f"modality_counts={modality_counts}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f"max_all_models_image_seconds={timing_document['max_all_models_image_seconds']:.3f}")
    print(f"cache={cache_path}")


if __name__ == "__main__":
    main()
