from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import environment_snapshot, load_yaml, resolve_path, sha256_file, utc_now, write_json
from src.data_contract import read_split_ids
from src.inference.labels import AIRCRAFT_CLASS_IDS, SHIP_CLASS_IDS, VEHICLE_CLASS_IDS
from src.inference.predictor import Predictor
from src.inference.runner import run_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen checkpoint on the fixed Data3 validation set.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/infer/exp006_eval_tiled.yaml")
    parser.add_argument("--device")
    parser.add_argument("--tile-batch", type=int)
    parser.add_argument("--ship-conf", type=float)
    parser.add_argument("--aircraft-conf", type=float)
    parser.add_argument("--vehicle-conf", type=float)
    parser.add_argument("--sample-count", type=int, default=0)
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    output_dir = resolve_path(args.output_dir)
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    for key in ("device", "tile_batch", "ship_conf", "aircraft_conf", "vehicle_conf"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    device = str(config.pop("device", "0"))
    ship_conf = float(config.pop("ship_conf"))
    aircraft_conf = float(config.pop("aircraft_conf"))
    vehicle_conf = float(config.pop("vehicle_conf"))
    thresholds = {
        **{category_id: ship_conf for category_id in SHIP_CLASS_IDS},
        **{category_id: aircraft_conf for category_id in AIRCRAFT_CLASS_IDS},
        **{category_id: vehicle_conf for category_id in VEHICLE_CLASS_IDS},
    }
    mode = str(config.pop("mode"))
    val_ids = read_split_ids(ROOT / "data_registry/splits/v1_scene_80_20/val.txt")
    if args.sample_count:
        if not 0 < args.sample_count <= len(val_ids):
            raise ValueError("sample-count must be between 1 and the validation size")
        if args.sample_count == 1:
            val_ids = [val_ids[0]]
        else:
            indexes = [round(index * (len(val_ids) - 1) / (args.sample_count - 1)) for index in range(args.sample_count)]
            val_ids = [val_ids[index] for index in indexes]
    image_root = ROOT / "workspace/data3_exp006/images/val"
    image_paths = [image_root / f"{stem}.jpg" for stem in val_ids]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prepared validation images missing; run script 02 first: {missing[:5]}")

    effective = {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device": device,
        "mode": mode,
        "class_thresholds": thresholds,
        "parameters": config,
        "image_count": len(image_paths),
        "started_at_utc": utc_now(),
        "environment": environment_snapshot(),
    }
    write_json(output_dir / "inference_manifest.json", effective)
    predictor = Predictor(
        model_path,
        device=device,
        mode=mode,
        class_thresholds=thresholds,
        **config,
    )
    summary = run_predictions(image_paths, predictor, output_dir)
    effective["finished_at_utc"] = utc_now()
    effective["result_sha256"] = sha256_file(summary.result_path)
    effective["timing_sha256"] = sha256_file(summary.timing_path)
    write_json(output_dir / "inference_manifest.json", effective)
    print(f"images={summary.image_count}")
    print(f"total_seconds={summary.total_seconds:.3f}")
    print(f"max_image_seconds={summary.max_image_seconds:.3f}")
    print(f"result={summary.result_path}")


if __name__ == "__main__":
    main()
