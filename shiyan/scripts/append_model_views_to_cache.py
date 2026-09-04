"""Append scale and horizontal-flip views of one model to an existing cache."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.scripts.cache_multi_model_predictions import file_sha256, object_payload  # noqa: E402
from shiyan.src.inference.predictor import Predictor  # noqa: E402
from shiyan.src.inference.runner import evenly_spaced_sample, read_image_list  # noqa: E402
from shiyan.src.inference.types import Detection  # noqa: E402


def parse_views(values: list[str]) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"view must use NAME=IMGSZ[,flip]: {raw}")
        name, config_text = raw.split("=", 1)
        name = name.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"invalid view name: {name}")
        if name in views:
            raise ValueError(f"duplicate view name: {name}")
        parts = [item.strip().lower() for item in config_text.split(",") if item.strip()]
        if not parts:
            raise ValueError(f"view has no image size: {raw}")
        imgsz = int(parts[0])
        if imgsz <= 0:
            raise ValueError(f"view image size must be positive: {raw}")
        options = set(parts[1:])
        unknown = options - {"flip"}
        if unknown:
            raise ValueError(f"unknown view options: {sorted(unknown)}")
        views[name] = {"imgsz": imgsz, "horizontal_flip": "flip" in options}
    return views


def unflip_detections(detections: list[Detection], width: int) -> list[Detection]:
    restored: list[Detection] = []
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        restored.append(
            Detection(
                category_id=item.category_id,
                score=item.score,
                bbox=(float(width) - x2, y1, float(width) - x1, y2),
            )
        )
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        help="repeat NAME=IMGSZ or NAME=IMGSZ,flip",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=0, help="0 means all cached images")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--tile-batch", type=int, default=4)
    args = parser.parse_args()

    base_cache_path = Path(args.base_cache)
    model_path = Path(args.model).resolve()
    if not base_cache_path.is_file():
        raise FileNotFoundError(f"base cache not found: {base_cache_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    views = parse_views(args.view)
    if args.sample_count < 0:
        raise ValueError("sample-count must be non-negative")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("conf and iou must be in [0, 1]")

    cache = json.loads(base_cache_path.read_text(encoding="utf-8"))
    if cache.get("schema_version") != "multi_model_cache_v1" or cache.get("status") != "success":
        raise ValueError("base cache does not satisfy multi_model_cache_v1")
    existing_names = set(cache["metadata"]["model_order"])
    collisions = existing_names.intersection(views)
    if collisions:
        raise ValueError(f"view names already exist in cache: {sorted(collisions)}")

    image_list_path = base_cache_path.parent / "image_list.txt"
    if not image_list_path.is_file():
        raise FileNotFoundError(f"base cache image list is missing: {image_list_path}")
    all_image_paths = read_image_list(image_list_path)
    if len(all_image_paths) != len(cache["images"]):
        raise ValueError("base cache image count disagrees with image list")
    for path, entry in zip(all_image_paths, cache["images"]):
        if path.name != entry["file_name"]:
            raise ValueError("base cache image order disagrees with image list")
    image_paths = (
        all_image_paths
        if args.sample_count == 0
        else evenly_spaced_sample(all_image_paths, args.sample_count)
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_cache = copy.deepcopy(cache)
    selected_names = {path.name for path in image_paths}
    output_cache["images"] = [
        entry for entry in output_cache["images"] if entry["file_name"] in selected_names
    ]
    output_cache["metadata"]["image_count"] = len(image_paths)
    model_hash = file_sha256(model_path)
    append_start = time.perf_counter()
    for view_name, view in views.items():
        print(f"view_start={view_name} config={view}", flush=True)
        predictor = Predictor(
            model_path,
            device=args.device,
            imgsz=int(view["imgsz"]),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            mode=args.mode,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            merge_iou=args.merge_iou,
            tile_batch=args.tile_batch,
        )
        view_start = time.perf_counter()
        for index, (image_path, entry) in enumerate(zip(image_paths, output_cache["images"]), start=1):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unable to read image: {image_path}")
            input_image = cv2.flip(image, 1) if view["horizontal_flip"] else image
            image_start = time.perf_counter()
            width, height, detections = predictor.predict_array(input_image)
            elapsed = time.perf_counter() - image_start
            if view["horizontal_flip"]:
                detections = unflip_detections(detections, width)
            if (width, height) != (int(entry["width"]), int(entry["height"])):
                raise ValueError(f"image dimensions disagree for {image_path.name}")
            entry["models"][view_name] = [object_payload(item) for item in detections]
            entry["seconds"][view_name] = round(elapsed, 6)
            if index % 100 == 0 or index == len(image_paths):
                print(f"view={view_name} completed={index}/{len(image_paths)}", flush=True)
        view_seconds = time.perf_counter() - view_start
        output_cache["metadata"]["models"][view_name] = {
            "path": str(model_path),
            "sha256": model_hash,
            "total_seconds": round(view_seconds, 6),
            "view": view,
        }
        output_cache["metadata"]["model_order"].append(view_name)
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    append_seconds = time.perf_counter() - append_start
    output_cache["metadata"]["appended_views"] = {
        "source_cache": str(base_cache_path),
        "model": str(model_path),
        "model_sha256": model_hash,
        "views": views,
        "append_wall_seconds": round(append_seconds, 6),
    }
    output_cache_path = output_dir / "raw_cache.json"
    output_cache_path.write_text(
        json.dumps(output_cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "image_list.txt").write_text(
        "\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8"
    )
    max_all_views = max(
        sum(float(value) for value in entry["seconds"].values())
        for entry in output_cache["images"]
    )
    summary = {
        "image_count": len(image_paths),
        "base_cache": str(base_cache_path),
        "appended_views": views,
        "append_wall_seconds": round(append_seconds, 6),
        "max_all_cached_views_image_seconds": round(max_all_views, 6),
    }
    (output_dir / "append_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"images={len(image_paths)}")
    print(f"views={list(views)}")
    print(f"append_seconds={append_seconds:.3f}")
    print(f"max_all_cached_views_image_seconds={max_all_views:.3f}")
    print(f"cache={output_cache_path}")


if __name__ == "__main__":
    main()
