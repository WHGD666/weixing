"""Local inference entrypoint with the official result.json output contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.src.inference.predictor import Predictor  # noqa: E402
from shiyan.src.inference.runner import collect_input_images, run_predictions  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to the frozen .pt model")
    parser.add_argument("--input", required=True, help="directory containing first-level input images")
    parser.add_argument("--output", required=True, help="directory for result.json and local timing files")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--mode", choices=("direct", "tiled"), default="direct")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--save-vis", action="store_true", help="save annotated images for local inspection")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_paths = collect_input_images(args.input)
    predictor = Predictor(
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
    summary = run_predictions(image_paths, predictor, args.output, save_visualizations=args.save_vis)
    print(f"images={summary.image_count}")
    print(f"total_seconds={summary.total_seconds:.3f}")
    print(f"max_image_seconds={summary.max_image_seconds:.3f}")
    print(f"result={summary.result_path}")
    print(f"timings={summary.timing_path}")


if __name__ == "__main__":
    main()
