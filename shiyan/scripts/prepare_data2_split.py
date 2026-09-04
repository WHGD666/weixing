#!/usr/bin/env python
"""Map frozen original splits to a sequentially renamed editable dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        old_name = row["old_image_name"]
        if old_name in by_name:
            raise SystemExit(f"Duplicate old image name in manifest: {old_name}")
        by_name[old_name] = row
    if len(by_name) != len(rows):
        raise SystemExit("Manifest mapping is not one-to-one")
    return by_name


def map_split(
    source: Path,
    destination: Path,
    mapping: dict[str, dict[str, str]],
    image_root: Path,
) -> int:
    mapped: list[str] = []
    for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
        source_name = Path(raw_line.strip().replace("\\", "/")).name
        if not source_name:
            continue
        row = mapping.get(source_name)
        if row is None:
            raise SystemExit(f"Split entry missing from rename manifest: {source_name}")
        mapped.append((image_root / row["new_image_name"]).as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(mapped) + "\n", encoding="utf-8")
    return len(mapped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="shiyan/data2")
    parser.add_argument(
        "--manifest",
        default="shiyan/data2/manifests/rename_0001_4481_manifest.csv",
    )
    parser.add_argument(
        "--source-split-root",
        default="shiyan/data_registry/split_assignments/v1_scene_80_20",
    )
    parser.add_argument(
        "--output-split-root",
        default="shiyan/data_registry/split_assignments/v1_scene_80_20_data2",
    )
    args = parser.parse_args()

    root = Path(args.root)
    manifest = read_manifest(Path(args.manifest))
    image_count = len(list((root / "images" / "train").glob("*.jpg")))
    if image_count != len(manifest):
        raise SystemExit(
            f"Image count does not match manifest: images={image_count} manifest={len(manifest)}"
        )

    source_root = Path(args.source_split_root)
    output_root = Path(args.output_split_root)
    image_root = root / "images" / "train"
    counts = {
        split: map_split(
            source_root / f"{split}.txt",
            output_root / f"{split}.txt",
            manifest,
            image_root,
        )
        for split in ("train", "val")
    }
    if counts["train"] + counts["val"] != len(manifest):
        raise SystemExit(f"Split counts do not cover all images: {counts}")

    metadata = {
        "source_protocol": "v1_scene_80_20",
        "dataset_root": str(root).replace("\\", "/"),
        "manifest": str(Path(args.manifest)).replace("\\", "/"),
        "mapping": "old image names mapped to sequential editable-dataset names",
        "image_count": image_count,
        "split_counts": counts,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
