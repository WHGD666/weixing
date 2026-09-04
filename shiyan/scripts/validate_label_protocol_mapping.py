"""Validate that original and manually revised protocols use identical validation images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_image_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    images = [Path(line.strip()).resolve() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not images:
        raise ValueError(f"image list is empty: {path}")
    missing = [image for image in images if not image.is_file()]
    if missing:
        raise FileNotFoundError(f"image list contains missing file: {missing[0]}")
    return images


def index_by_file_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        name = Path(row["path"]).name
        if name in indexed:
            raise ValueError(f"duplicate manifest file name: {name}")
        indexed[name] = row
    return indexed


def optional_int(value: str | None) -> int:
    return int(value) if value not in (None, "") else 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def read_yolo_objects(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    objects: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        values = raw_line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"invalid YOLO label at {path}:{line_number}")
        objects.append((int(values[0]), tuple(float(value) for value in values[1:])))
    return objects


def normalized_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x, left_y, left_w, left_h = left
    right_x, right_y, right_w, right_h = right
    left_box = (left_x - left_w / 2, left_y - left_h / 2, left_x + left_w / 2, left_y + left_h / 2)
    right_box = (
        right_x - right_w / 2,
        right_y - right_h / 2,
        right_x + right_w / 2,
        right_y + right_h / 2,
    )
    x1 = max(left_box[0], right_box[0])
    y1 = max(left_box[1], right_box[1])
    x2 = min(left_box[2], right_box[2])
    y2 = min(left_box[3], right_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left_box[2] - left_box[0]) * max(0.0, left_box[3] - left_box[1])
    right_area = max(0.0, right_box[2] - right_box[0]) * max(0.0, right_box[3] - right_box[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def same_class_match_ious(
    original: list[tuple[int, tuple[float, float, float, float]]],
    revised: list[tuple[int, tuple[float, float, float, float]]],
) -> list[float]:
    pairs = sorted(
        (
            (normalized_iou(old_box, new_box), old_index, new_index)
            for old_index, (old_class, old_box) in enumerate(original)
            for new_index, (new_class, new_box) in enumerate(revised)
            if old_class == new_class
        ),
        reverse=True,
    )
    used_old: set[int] = set()
    used_new: set[int] = set()
    matches: list[float] = []
    for overlap, old_index, new_index in pairs:
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        matches.append(overlap)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-image-list", required=True)
    parser.add_argument("--revised-image-list", required=True)
    parser.add_argument("--rename-manifest", required=True)
    parser.add_argument("--original-image-manifest", required=True)
    parser.add_argument("--revised-image-manifest", required=True)
    parser.add_argument("--original-label-manifest", required=True)
    parser.add_argument("--revised-label-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original_paths = read_image_list(Path(args.original_image_list))
    revised_paths = read_image_list(Path(args.revised_image_list))
    rename_rows = read_csv(Path(args.rename_manifest))
    rename_by_old: dict[str, dict[str, str]] = {}
    for row in rename_rows:
        old_name = row["old_image_name"]
        if old_name in rename_by_old:
            raise ValueError(f"duplicate old image name in rename manifest: {old_name}")
        rename_by_old[old_name] = row

    original_images = index_by_file_name(read_csv(Path(args.original_image_manifest)))
    revised_images = index_by_file_name(read_csv(Path(args.revised_image_manifest)))
    original_labels = index_by_file_name(read_csv(Path(args.original_label_manifest)))
    revised_labels = index_by_file_name(read_csv(Path(args.revised_label_manifest)))
    revised_by_name = {path.name: path for path in revised_paths}

    if len(original_paths) != len(revised_paths):
        raise ValueError(
            f"validation list counts differ: original={len(original_paths)} revised={len(revised_paths)}"
        )

    detail_rows: list[dict[str, Any]] = []
    mapped_names: list[str] = []
    image_hash_mismatches = 0
    dimension_mismatches = 0
    changed_labels = 0
    numeric_changed_labels = 0
    object_count_changed_files = 0
    class_histogram_changed_files = 0
    equivalent_files = {"iou_0_50": 0, "iou_0_90": 0, "iou_0_99": 0}
    original_manifest_count_mismatches = 0
    revised_manifest_count_mismatches = 0
    old_object_count = 0
    revised_object_count = 0
    for original_path in original_paths:
        rename = rename_by_old.get(original_path.name)
        if rename is None:
            raise ValueError(f"original validation image missing from rename manifest: {original_path.name}")
        revised_name = rename["new_image_name"]
        mapped_names.append(revised_name)
        if revised_name not in revised_by_name:
            raise ValueError(f"mapped revised image is not in revised validation list: {revised_name}")
        old_image = original_images[original_path.name]
        new_image = revised_images[revised_name]
        image_hash_equal = old_image["sha256"].lower() == new_image["sha256"].lower()
        dimensions_equal = (old_image["width"], old_image["height"]) == (
            new_image["width"],
            new_image["height"],
        )
        image_hash_mismatches += int(not image_hash_equal)
        dimension_mismatches += int(not dimensions_equal)

        old_label_name = original_path.with_suffix(".txt").name
        new_label_name = Path(revised_name).with_suffix(".txt").name
        old_label = original_labels[old_label_name]
        new_label = revised_labels[new_label_name]
        old_label_path = Path(old_label["path"])
        new_label_path = Path(new_label["path"])
        old_actual_sha256 = file_sha256(old_label_path)
        new_actual_sha256 = file_sha256(new_label_path)
        label_changed = old_actual_sha256 != new_actual_sha256
        changed_labels += int(label_changed)
        old_values = read_yolo_objects(old_label_path)
        new_values = read_yolo_objects(new_label_path)
        numeric_equal = sorted(old_values) == sorted(new_values)
        numeric_changed_labels += int(not numeric_equal)
        class_histogram_equal = Counter(item[0] for item in old_values) == Counter(
            item[0] for item in new_values
        )
        class_histogram_changed_files += int(not class_histogram_equal)
        object_count_changed_files += int(len(old_values) != len(new_values))
        match_ious = same_class_match_ious(old_values, new_values)
        old_manifest_objects = optional_int(old_label.get("object_count"))
        new_manifest_objects = optional_int(new_label.get("object_count"))
        old_objects = len(old_values)
        new_objects = len(new_values)
        original_manifest_count_mismatches += int(old_manifest_objects != old_objects)
        revised_manifest_count_mismatches += int(new_manifest_objects != new_objects)
        old_object_count += old_objects
        revised_object_count += new_objects
        equivalent: dict[str, bool] = {}
        for label, threshold in (("iou_0_50", 0.50), ("iou_0_90", 0.90), ("iou_0_99", 0.99)):
            matched_count = sum(overlap >= threshold for overlap in match_ious)
            equivalent[label] = (
                len(old_values) == len(new_values)
                and matched_count == len(old_values)
            )
            equivalent_files[label] += int(equivalent[label])
        detail_rows.append(
            {
                "original_image_name": original_path.name,
                "revised_image_name": revised_name,
                "image_sha256_equal": image_hash_equal,
                "dimensions_equal": dimensions_equal,
                "original_label_sha256": old_actual_sha256,
                "revised_label_sha256": new_actual_sha256,
                "label_changed": label_changed,
                "numeric_equal": numeric_equal,
                "class_histogram_equal": class_histogram_equal,
                "equivalent_iou_0_50": equivalent["iou_0_50"],
                "equivalent_iou_0_90": equivalent["iou_0_90"],
                "equivalent_iou_0_99": equivalent["iou_0_99"],
                "minimum_matched_iou": min(match_ious) if match_ious else None,
                "original_objects": old_objects,
                "revised_objects": new_objects,
                "object_delta": new_objects - old_objects,
                "original_manifest_objects": old_manifest_objects,
                "revised_manifest_objects": new_manifest_objects,
            }
        )

    actual_revised_names = [path.name for path in revised_paths]
    order_equal = mapped_names == actual_revised_names
    unique_mapping = len(set(mapped_names)) == len(mapped_names)
    ok = (
        order_equal
        and unique_mapping
        and image_hash_mismatches == 0
        and dimension_mismatches == 0
    )
    summary = {
        "ok": ok,
        "original_protocol": str(Path(args.original_image_list)),
        "revised_protocol": str(Path(args.revised_image_list)),
        "rename_manifest": str(Path(args.rename_manifest)),
        "image_count": len(original_paths),
        "order_equal": order_equal,
        "unique_mapping": unique_mapping,
        "image_hash_mismatches": image_hash_mismatches,
        "dimension_mismatches": dimension_mismatches,
        "changed_label_files": changed_labels,
        "unchanged_label_files": len(original_paths) - changed_labels,
        "numeric_changed_label_files": numeric_changed_labels,
        "object_count_changed_files": object_count_changed_files,
        "class_histogram_changed_files": class_histogram_changed_files,
        "equivalent_label_files": equivalent_files,
        "original_manifest_count_mismatches": original_manifest_count_mismatches,
        "revised_manifest_count_mismatches": revised_manifest_count_mismatches,
        "label_manifests_current": (
            original_manifest_count_mismatches == 0 and revised_manifest_count_mismatches == 0
        ),
        "original_object_count": old_object_count,
        "revised_object_count": revised_object_count,
        "object_count_delta": revised_object_count - old_object_count,
    }
    with (output_dir / "protocol_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    (output_dir / "protocol_mapping.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    if not ok:
        raise SystemExit("protocol mapping validation failed")


if __name__ == "__main__":
    main()
