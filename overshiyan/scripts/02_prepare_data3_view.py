from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml, write_json
from src.data_contract import (
    audit_dataset,
    load_class_names,
    prepare_dataset_view,
    read_split_ids,
)


def _safe_remove(path: Path) -> None:
    workspace = (ROOT / "workspace").resolve()
    resolved = path.resolve()
    if workspace not in resolved.parents:
        raise ValueError(f"Refusing to remove path outside workspace: {resolved}")
    shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-destructive Data3 train/val view.")
    parser.add_argument("--source", default="data3")
    parser.add_argument("--output", default="workspace/data3_exp006")
    parser.add_argument("--image-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = (ROOT / args.source).resolve()
    output = (ROOT / args.output).resolve()
    contract = load_yaml(ROOT / "data_registry/contracts/task_contract.yaml")
    class_names = load_class_names(ROOT / "data_registry/contracts/class_names.txt")
    train_ids = read_split_ids(ROOT / "data_registry/splits/v1_scene_80_20/train.txt")
    val_ids = read_split_ids(ROOT / "data_registry/splits/v1_scene_80_20/val.txt")
    if len(train_ids) != int(contract["train_count"]) or len(val_ids) != int(contract["val_count"]):
        raise ValueError(f"Unexpected split sizes: train={len(train_ids)} val={len(val_ids)}")
    if set(train_ids) & set(val_ids):
        raise ValueError("Train and validation IDs overlap")
    if len(set(train_ids) | set(val_ids)) != int(contract["expected_images"]):
        raise ValueError("Split IDs do not cover the complete dataset")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Prepared view already exists: {output}; use --force to rebuild")
        _safe_remove(output)

    source_audit_before = audit_dataset(
        source / "images/train", source / "labels/train", class_names
    )

    report = prepare_dataset_view(
        image_root=source / "images/train",
        label_root=source / "labels/train",
        output_root=output,
        train_ids=train_ids,
        val_ids=val_ids,
        class_names=class_names,
        image_mode=args.image_mode,
    )
    prepared_audits = {}
    for split, expected_count in (("train", len(train_ids)), ("val", len(val_ids))):
        audit = audit_dataset(
            output / "images" / split,
            output / "labels" / split,
            class_names,
        )
        if audit["image_count"] != expected_count or audit["label_count"] != expected_count:
            raise ValueError(f"prepared {split} count mismatch")
        if audit["missing_label_stems"] or audit["missing_image_stems"] or audit["invalid_labels"]:
            raise ValueError(f"prepared {split} audit failed")
        if audit["duplicate_rows"]:
            raise ValueError(f"prepared {split} still contains exact duplicate label rows")
        prepared_audits[split] = audit
        write_json(output / f"{split}_audit.json", audit)
    source_audit_after = audit_dataset(
        source / "images/train", source / "labels/train", class_names
    )
    if source_audit_before["fingerprint_sha256"] != source_audit_after["fingerprint_sha256"]:
        raise RuntimeError("source Data3 fingerprint changed during preparation")
    report["source_fingerprint_before"] = source_audit_before["fingerprint_sha256"]
    report["source_fingerprint_after"] = source_audit_after["fingerprint_sha256"]
    report["prepared_audit_fingerprints"] = {
        split: audit["fingerprint_sha256"] for split, audit in prepared_audits.items()
    }
    write_json(output / "prepared_dataset.json", report)
    print(
        f"ok=true train={report['split_counts']['train']} val={report['split_counts']['val']} "
        f"removed_exact_duplicates={report['removed_exact_duplicate_rows']}"
    )
    print(f"data={report['data_yaml']} manifest={report['manifest']}")


if __name__ == "__main__":
    main()
