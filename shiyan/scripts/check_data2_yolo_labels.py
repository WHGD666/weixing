#!/usr/bin/env python
"""Validate editable data2 YOLO labels and class metadata."""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_CLASSES = [
    "HM",
    "LQS",
    "QHS",
    "MS",
    "A1_SU-35",
    "A2_C-130",
    "A3_C-17",
    "A4_C-5",
    "A5_F-16",
    "A6_TU-160",
    "A7_E-3",
    "A8_B-52",
    "A9_P-3C",
    "A10_B-1B",
    "A11_E-8",
    "A12_TU-22",
    "A13_F-15",
    "A14_KC-135",
    "A15_F-22",
    "A16_FA-18",
    "A17_TU-95",
    "A18_KC-10",
    "A19_SU-34",
    "A20_SU-24",
    "FSC",
]


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="shiyan/data2")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    class_file = labels_dir / "classes.txt"

    if not class_file.exists():
        raise SystemExit(f"Missing classes.txt: {class_file}")
    actual_classes = read_classes(class_file)
    if actual_classes != EXPECTED_CLASSES:
        raise SystemExit(f"classes.txt mismatch: expected {len(EXPECTED_CLASSES)} fixed classes, got {len(actual_classes)}")

    image_stems = {path.stem for path in images_dir.glob("*.jpg")}
    label_files = [path for path in labels_dir.glob("*.txt") if path.name != "classes.txt"]
    label_stems = {path.stem for path in label_files}

    missing = sorted(image_stems - label_stems)
    orphan = sorted(label_stems - image_stems)
    if missing:
        raise SystemExit(f"Images without labels: {missing[:10]}")
    if orphan:
        raise SystemExit(f"Labels without images: {orphan[:10]}")

    bad: list[str] = []
    for path in sorted(label_files):
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad.append(f"{path.name}:{line_number}: expected 5 columns, got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                bad.append(f"{path.name}:{line_number}: non-numeric YOLO row")
                continue
            if cls < 0 or cls >= len(EXPECTED_CLASSES):
                bad.append(f"{path.name}:{line_number}: class id out of range: {cls}")
            if any(value < 0 or value > 1 for value in coords):
                bad.append(f"{path.name}:{line_number}: coordinate out of range 0..1")

    if bad:
        raise SystemExit("\n".join(bad[:30]))

    print(f"ok=true images={len(image_stems)} labels={len(label_files)} classes={len(actual_classes)}")


if __name__ == "__main__":
    main()
