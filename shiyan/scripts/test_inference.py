"""Run a deterministic validation-image smoke test and validate result.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.src.inference.predictor import Predictor  # noqa: E402
from shiyan.src.inference.runner import evenly_spaced_sample, read_image_list, run_predictions  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-list", required=True, help="fixed validation image list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=12)
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
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    all_paths = read_image_list(args.image_list)
    image_paths = evenly_spaced_sample(all_paths, args.sample_count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_list_path = output_dir / "image_list.txt"
    selected_list_path.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
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
    summary = run_predictions(image_paths, predictor, output_dir, save_visualizations=args.save_vis)
    document = json.loads(summary.result_path.read_text(encoding="utf-8"))
    validate_result_document(document, [path.name for path in image_paths])
    object_count = sum(len(item["objects"]) for item in document["images"])
    print(f"validated_images={summary.image_count}")
    print(f"validated_objects={object_count}")
    print(f"total_seconds={summary.total_seconds:.3f}")
    print(f"max_image_seconds={summary.max_image_seconds:.3f}")
    print(f"result={summary.result_path}")
    print(f"image_list={selected_list_path}")


if __name__ == "__main__":
    main()
