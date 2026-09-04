from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml, read_nonempty_lines, write_json
from src.data_contract import audit_dataset, load_class_names, validate_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable Data3 YOLO images and labels.")
    parser.add_argument("--root", default="data3")
    parser.add_argument("--output", default="data_registry/audits/data3_source_audit.json")
    parser.add_argument("--strict", action="store_true", help="Decode every image and fail on contract errors.")
    parser.add_argument("--hash-image-contents", action="store_true")
    args = parser.parse_args()

    dataset_root = (ROOT / args.root).resolve()
    contract = load_yaml(ROOT / "data_registry/contracts/task_contract.yaml")
    class_names = load_class_names(ROOT / "data_registry/contracts/class_names.txt")
    report = audit_dataset(
        dataset_root / "images/train",
        dataset_root / "labels/train",
        class_names,
        decode_images=args.strict,
        hash_image_contents=args.hash_image_contents,
    )
    errors = validate_audit(report, contract)
    source_classes = read_nonempty_lines(dataset_root / "labels/train/classes.txt")
    if source_classes != class_names:
        errors.append("data3 labels/train/classes.txt does not match the frozen class contract")
    source_yaml = load_yaml(dataset_root / "coco.yaml")
    yaml_names = source_yaml.get("names")
    if isinstance(yaml_names, dict):
        normalized_yaml_names = [str(yaml_names[index]) for index in range(len(yaml_names))]
    elif isinstance(yaml_names, list):
        normalized_yaml_names = [str(value) for value in yaml_names]
    else:
        normalized_yaml_names = []
    if normalized_yaml_names != class_names:
        errors.append("data3 coco.yaml does not match the frozen class contract")
    report["source_class_contract"] = {
        "classes_txt_matches": source_classes == class_names,
        "coco_yaml_matches": normalized_yaml_names == class_names,
    }
    report["contract_errors"] = errors
    report["ok"] = not errors
    output = (ROOT / args.output).resolve()
    write_json(output, report)
    print(
        f"ok={report['ok']} images={report['image_count']} labels={report['label_count']} "
        f"objects={report['object_count']} duplicates={sum(report['duplicate_rows'].values())} "
        f"boundary_crossing={sum(report['boundary_crossing_rows'].values())}"
    )
    print(f"empty_labels={report['empty_label_stems']}")
    print(f"fingerprint={report['fingerprint_sha256']} output={output}")
    for error in errors:
        print(f"ERROR: {error}")
    if args.strict and errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
