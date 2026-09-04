from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml, resolve_path, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the prepared dataset through Ultralytics itself.")
    parser.add_argument("--data", default="workspace/data3_exp006/data.yaml")
    parser.add_argument("--train-config", default="configs/train/exp006_x0_data3_yolo11x_1024.yaml")
    parser.add_argument("--output", default="data_registry/audits/data3_ultralytics_loader.json")
    args = parser.parse_args()

    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    train_config = load_yaml(resolve_path(args.train_config))
    overrides = {key: value for key, value in train_config.items() if key in DEFAULT_CFG_DICT}
    config = get_cfg(overrides=overrides)
    data = check_det_dataset(str(resolve_path(args.data)))
    contract = load_yaml(ROOT / "data_registry/contracts/task_contract.yaml")
    report = {
        "created_at_utc": utc_now(),
        "data": str(resolve_path(args.data)),
        "class_count": int(data["nc"]),
        "splits": {},
        "errors": [],
    }
    for mode in ("train", "val"):
        dataset = build_yolo_dataset(
            config,
            data[mode],
            batch=int(train_config["batch"]),
            data=data,
            mode=mode,
            rect=mode == "val",
            stride=32,
        )
        object_count = sum(len(item["cls"]) for item in dataset.labels)
        empty_count = sum(len(item["cls"]) == 0 for item in dataset.labels)
        report["splits"][mode] = {
            "images": len(dataset),
            "objects": object_count,
            "empty_images": empty_count,
        }
        expected_images = int(contract[f"{mode}_count"])
        expected_objects = int(contract[f"prepared_{mode}_objects"])
        expected_empty = int(contract[f"prepared_{mode}_empty_images"])
        if (len(dataset), object_count, empty_count) != (
            expected_images,
            expected_objects,
            expected_empty,
        ):
            report["errors"].append(
                f"{mode}: expected {(expected_images, expected_objects, expected_empty)}, "
                f"got {(len(dataset), object_count, empty_count)}"
            )
    if int(data["nc"]) != int(contract["expected_classes"]):
        report["errors"].append(f"expected {contract['expected_classes']} classes, got {data['nc']}")
    report["ok"] = not report["errors"]
    output = resolve_path(args.output)
    write_json(output, report)
    print(f"ok={report['ok']} splits={report['splits']} output={output}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
