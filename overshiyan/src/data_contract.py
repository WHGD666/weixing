from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import read_nonempty_lines, sha256_file, utc_now, write_json, write_yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ParsedLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def canonical(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.10g} {self.y_center:.10g} "
            f"{self.width:.10g} {self.height:.10g}"
        )


def load_class_names(path: Path) -> list[str]:
    names = read_nonempty_lines(path)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate class names in {path}")
    return names


def parse_label_file(path: Path, class_count: int) -> tuple[list[ParsedLabel], list[str]]:
    parsed: list[ParsedLabel] = []
    errors: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) != 5:
            errors.append(f"{path.name}:{line_number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            class_value = float(fields[0])
            class_id = int(class_value)
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path.name}:{line_number}: non-numeric value")
            continue
        if class_value != class_id or not 0 <= class_id < class_count:
            errors.append(f"{path.name}:{line_number}: invalid class id {fields[0]}")
            continue
        x_center, y_center, width, height = coords
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            errors.append(f"{path.name}:{line_number}: center outside [0, 1]")
            continue
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            errors.append(f"{path.name}:{line_number}: width/height outside (0, 1]")
            continue
        parsed.append(ParsedLabel(class_id, x_center, y_center, width, height))
    return parsed, errors


def _image_files(root: Path) -> dict[str, Path]:
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    stems: dict[str, Path] = {}
    for path in files:
        if path.stem in stems:
            raise ValueError(f"Duplicate image stem: {path.stem}")
        stems[path.stem] = path
    return stems


def _label_files(root: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in root.glob("*.txt")
        if path.name.lower() != "classes.txt"
    }


def audit_dataset(
    image_root: Path,
    label_root: Path,
    class_names: list[str],
    *,
    decode_images: bool = False,
    hash_image_contents: bool = False,
) -> dict[str, Any]:
    images = _image_files(image_root)
    labels = _label_files(label_root)
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    empty_stems: list[str] = []
    invalid_labels: list[str] = []
    duplicate_rows: dict[str, int] = {}
    boundary_crossing_rows: dict[str, int] = {}
    class_counts: Counter[int] = Counter()
    label_hash_entries: list[str] = []
    image_hash_entries: list[str] = []
    unreadable_images: list[str] = []

    for stem in sorted(labels):
        path = labels[stem]
        rows, errors = parse_label_file(path, len(class_names))
        invalid_labels.extend(errors)
        if not rows and path.stat().st_size == 0:
            empty_stems.append(stem)
        canonical = [row.canonical() for row in rows]
        duplicate_count = len(canonical) - len(set(canonical))
        if duplicate_count:
            duplicate_rows[stem] = duplicate_count
        boundary_count = sum(
            row.x_center - row.width / 2 < 0.0
            or row.x_center + row.width / 2 > 1.0
            or row.y_center - row.height / 2 < 0.0
            or row.y_center + row.height / 2 > 1.0
            for row in rows
        )
        if boundary_count:
            boundary_crossing_rows[stem] = boundary_count
        class_counts.update(row.class_id for row in rows)
        label_hash_entries.append(f"{path.name}:{sha256_file(path)}")

    cv2 = None
    if decode_images:
        try:
            import cv2 as cv2_module

            cv2 = cv2_module
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for --decode-images") from exc

    for stem in sorted(images):
        path = images[stem]
        if hash_image_contents:
            signature = sha256_file(path)
        else:
            signature = str(path.stat().st_size)
        image_hash_entries.append(f"{path.name}:{signature}")
        if cv2 is not None and cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
            unreadable_images.append(path.name)

    fingerprint = hashlib.sha256()
    for value in [*label_hash_entries, *image_hash_entries]:
        fingerprint.update(value.encode("utf-8"))
        fingerprint.update(b"\n")

    return {
        "created_at_utc": utc_now(),
        "image_root": str(image_root),
        "label_root": str(label_root),
        "image_count": len(images),
        "label_count": len(labels),
        "object_count": sum(class_counts.values()),
        "class_count": len(class_names),
        "class_counts": {str(index): class_counts[index] for index in range(len(class_names))},
        "class_names": class_names,
        "empty_label_stems": empty_stems,
        "missing_label_stems": missing_labels,
        "missing_image_stems": missing_images,
        "invalid_labels": invalid_labels,
        "duplicate_rows": duplicate_rows,
        "boundary_crossing_rows": boundary_crossing_rows,
        "unreadable_images": unreadable_images,
        "image_content_hashed": hash_image_contents,
        "fingerprint_sha256": fingerprint.hexdigest(),
    }


def validate_audit(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "image_count": int(contract["expected_images"]),
        "label_count": int(contract["expected_labels"]),
        "class_count": int(contract["expected_classes"]),
    }
    for key, value in expected.items():
        if report[key] != value:
            errors.append(f"{key}: expected {value}, got {report[key]}")
    for key in ("missing_label_stems", "missing_image_stems", "invalid_labels", "unreadable_images"):
        if report[key]:
            errors.append(f"{key}: {report[key][:10]}")
    expected_empty = sorted(str(value) for value in contract["intentional_empty_stems"])
    if sorted(report["empty_label_stems"]) != expected_empty:
        errors.append(
            f"empty_label_stems: expected {expected_empty}, got {sorted(report['empty_label_stems'])}"
        )
    return errors


def read_split_ids(path: Path) -> list[str]:
    ids = read_nonempty_lines(path)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in split file: {path}")
    for stem in ids:
        if Path(stem).name != stem or Path(stem).suffix:
            raise ValueError(f"Split entries must be bare stems, got {stem!r} in {path}")
    return ids


def deduplicated_label_text(path: Path, class_count: int) -> tuple[str, int, int]:
    rows, errors = parse_label_file(path, class_count)
    if errors:
        raise ValueError("; ".join(errors))
    unique: list[str] = []
    seen: set[str] = set()
    for row in rows:
        line = row.canonical()
        if line not in seen:
            unique.append(line)
            seen.add(line)
    text = "" if not unique else "\n".join(unique) + "\n"
    return text, len(rows), len(rows) - len(unique)


def materialize_image(source: Path, destination: Path, mode: str = "copy") -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "hardlink":
        os.link(source, destination)
        return "hardlink"
    raise ValueError(f"unsupported image materialization mode: {mode}")


def prepare_dataset_view(
    *,
    image_root: Path,
    label_root: Path,
    output_root: Path,
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    class_names: list[str],
    image_mode: str = "copy",
) -> dict[str, Any]:
    image_map = _image_files(image_root)
    label_map = _label_files(label_root)
    rows: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    removed_duplicates = 0
    split_counts: Counter[str] = Counter()

    for split, ids in (("train", list(train_ids)), ("val", list(val_ids))):
        for stem in ids:
            if stem not in image_map or stem not in label_map:
                raise FileNotFoundError(f"Missing source pair for {stem}")
            source_image = image_map[stem]
            source_label = label_map[stem]
            target_image = output_root / "images" / split / source_image.name
            target_label = output_root / "labels" / split / f"{stem}.txt"
            link_method = materialize_image(source_image, target_image, image_mode)
            label_text, source_rows, duplicate_count = deduplicated_label_text(
                source_label, len(class_names)
            )
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text(label_text, encoding="utf-8", newline="\n")
            method_counts[link_method] += 1
            removed_duplicates += duplicate_count
            split_counts[split] += 1
            rows.append(
                {
                    "stem": stem,
                    "split": split,
                    "source_image": str(source_image),
                    "source_label": str(source_label),
                    "prepared_image": str(target_image),
                    "prepared_label": str(target_label),
                    "image_method": link_method,
                    "source_rows": source_rows,
                    "removed_exact_duplicate_rows": duplicate_count,
                    "source_label_sha256": sha256_file(source_label),
                    "prepared_label_sha256": sha256_file(target_label),
                }
            )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    write_yaml(output_root / "data.yaml", data_yaml)
    report = {
        "created_at_utc": utc_now(),
        "output_root": str(output_root.resolve()),
        "split_counts": dict(split_counts),
        "image_materialization": dict(method_counts),
        "removed_exact_duplicate_rows": removed_duplicates,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "data_yaml": str((output_root / "data.yaml").resolve()),
        "source_data_unchanged": True,
    }
    write_json(output_root / "prepared_dataset.json", report)
    return report
