from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from inference.labels import AIRCRAFT_CLASS_IDS, SHIP_CLASS_IDS, VEHICLE_CLASS_IDS
from inference.predictor import Predictor
from inference.runner import collect_input_images, run_predictions


DEFAULT_MODEL = Path("/app/models/best.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Competition detector entry point.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det-per-tile", type=int, default=300)
    parser.add_argument("--max-det-image", type=int, default=3000)
    parser.add_argument("--mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--tile-batch", type=int, default=1)
    parser.add_argument("--ship-conf", type=float, default=0.30)
    parser.add_argument("--aircraft-conf", type=float, default=0.30)
    parser.add_argument("--vehicle-conf", type=float, default=0.35)
    parser.add_argument("--no-half", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    thresholds = {
        **{category_id: args.ship_conf for category_id in SHIP_CLASS_IDS},
        **{category_id: args.aircraft_conf for category_id in AIRCRAFT_CLASS_IDS},
        **{category_id: args.vehicle_conf for category_id in VEHICLE_CLASS_IDS},
    }
    predictor = Predictor(
        args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det_per_tile=args.max_det_per_tile,
        max_det_image=args.max_det_image,
        mode=args.mode,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        merge_iou=args.merge_iou,
        tile_batch=args.tile_batch,
        half=not args.no_half,
        class_thresholds=thresholds,
    )
    images = collect_input_images(args.input)
    summary = run_predictions(images, predictor, args.output)
    print(f"images={summary.image_count}", flush=True)
    print(f"total_seconds={summary.total_seconds:.3f}", flush=True)
    print(f"max_image_seconds={summary.max_image_seconds:.3f}", flush=True)
    print(f"result={summary.result_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fatal_error={exc}", file=sys.stderr, flush=True)
        raise
