#!/usr/bin/env python
"""Audit the original YOLO dataset without modifying source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


EXPECTED_CLASSES = {
    0: ("HM", "ship"),
    1: ("LQS", "ship"),
    2: ("QHS", "ship"),
    3: ("MS", "ship"),
    4: ("A1_SU-35", "aircraft"),
    5: ("A2_C-130", "aircraft"),
    6: ("A3_C-17", "aircraft"),
    7: ("A4_C-5", "aircraft"),
    8: ("A5_F-16", "aircraft"),
    9: ("A6_TU-160", "aircraft"),
    10: ("A7_E-3", "aircraft"),
    11: ("A8_B-52", "aircraft"),
    12: ("A9_P-3C", "aircraft"),
    13: ("A10_B-1B", "aircraft"),
    14: ("A11_E-8", "aircraft"),
    15: ("A12_TU-22", "aircraft"),
    16: ("A13_F-15", "aircraft"),
    17: ("A14_KC-135", "aircraft"),
    18: ("A15_F-22", "aircraft"),
    19: ("A16_FA-18", "aircraft"),
    20: ("A17_TU-95", "aircraft"),
    21: ("A18_KC-10", "aircraft"),
    22: ("A19_SU-34", "aircraft"),
    23: ("A20_SU-24", "aircraft"),
    24: ("FSC", "vehicle"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
YOLO_IGNORE_FILES = {"classes.txt"}
SCENE_CROP_RE = re.compile(r"^(?P<scene>.+?)_crop\d+$", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def scene_id(stem: str) -> str:
    match = SCENE_CROP_RE.match(stem)
    return match.group("scene") if match else stem


def read_yaml_names(dataset_yaml: Path) -> dict[int, str]:
    if not dataset_yaml.exists():
        return {}
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return {}


def bucket_long_side(width: int, height: int) -> str:
    long_side = max(width, height)
    if long_side <= 640:
        return "<=640"
    if long_side <= 1024:
        return "641-1024"
    if long_side <= 1536:
        return "1025-1536"
    if long_side <= 2048:
        return "1537-2048"
    if long_side <= 4096:
        return "2049-4096"
    if long_side <= 8192:
        return "4097-8192"
    return ">8192"


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    issue_type: str,
    path: str,
    message: str,
    split: str = "",
    line_no: int | str = "",
    class_id: int | str = "",
    value: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "issue_type": issue_type,
            "split": split,
            "path": path,
            "line_no": line_no,
            "class_id": class_id,
            "value": value,
            "message": message,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def markdown_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in shown:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"| ... | 仅展示前 {max_rows} 行，共 {len(rows)} 行 |" + " |" * (len(fields) - 2))
    return "\n".join(lines)


def audit(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = (Path(args.output_root).resolve() / "audits" / args.version)
    manifest_dir = Path(args.output_root).resolve() / "manifests" / args.version
    fingerprint_dir = Path(args.output_root).resolve() / "fingerprints"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_dir.mkdir(parents=True, exist_ok=True)

    images_root = data_root / "images"
    labels_root = data_root / "labels"
    dataset_yaml = data_root / "dataset.yaml"

    issues: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    label_file_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    dataset_names = read_yaml_names(dataset_yaml)
    class_contract_rows = []
    for class_id, (name, group) in EXPECTED_CLASSES.items():
        yaml_name = dataset_names.get(class_id, "")
        status = "ok" if yaml_name == name else "mismatch"
        class_contract_rows.append(
            {
                "class_id": class_id,
                "expected_name": name,
                "yaml_name": yaml_name,
                "group": group,
                "status": status,
            }
        )
        if status != "ok":
            add_issue(
                issues,
                "error",
                "class_mapping_mismatch",
                rel(dataset_yaml, repo_root) if dataset_yaml.exists() else "dataset.yaml",
                f"class {class_id} expected {name}, got {yaml_name}",
                class_id=class_id,
            )

    if len(dataset_names) != len(EXPECTED_CLASSES):
        add_issue(
            issues,
            "error",
            "class_count_mismatch",
            rel(dataset_yaml, repo_root) if dataset_yaml.exists() else "dataset.yaml",
            f"expected {len(EXPECTED_CLASSES)} classes, got {len(dataset_names)}",
            value=str(len(dataset_names)),
        )

    image_paths = sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raw_label_paths = sorted(p for p in labels_root.rglob("*.txt") if p.is_file())
    label_paths = [p for p in raw_label_paths if p.name not in YOLO_IGNORE_FILES]
    ignored_label_paths = [p for p in raw_label_paths if p.name in YOLO_IGNORE_FILES]

    image_by_split_stem: dict[tuple[str, str], Path] = {}
    label_by_split_stem: dict[tuple[str, str], Path] = {}
    image_hash_to_paths: dict[str, list[str]] = defaultdict(list)

    for path in image_paths:
        split = path.parent.relative_to(images_root).parts[0] if path.parent != images_root else ""
        key = (split, path.stem)
        if key in image_by_split_stem:
            add_issue(
                issues,
                "error",
                "duplicate_image_stem_in_split",
                rel(path, repo_root),
                f"duplicate image stem {path.stem} in split {split}",
                split=split,
            )
        image_by_split_stem[key] = path

    for path in label_paths:
        split = path.parent.relative_to(labels_root).parts[0] if path.parent != labels_root else ""
        key = (split, path.stem)
        if key in label_by_split_stem:
            add_issue(
                issues,
                "error",
                "duplicate_label_stem_in_split",
                rel(path, repo_root),
                f"duplicate label stem {path.stem} in split {split}",
                split=split,
            )
        label_by_split_stem[key] = path

    for path in ignored_label_paths:
        label_file_rows.append(
            {
                "split": path.parent.relative_to(labels_root).parts[0] if path.parent != labels_root else "",
                "path": rel(path, repo_root),
                "stem": path.stem,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "line_count": "",
                "object_count": "",
                "status": "ignored_auxiliary_file",
            }
        )

    for key, image_path in image_by_split_stem.items():
        split, stem = key
        image_rel = rel(image_path, repo_root)
        image_status = "ok"
        width = height = ""
        mode = ""
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
        except Exception as exc:  # noqa: BLE001
            image_status = "image_open_error"
            add_issue(issues, "error", "image_open_error", image_rel, str(exc), split=split)

        image_hash = sha256_file(image_path)
        image_hash_to_paths[image_hash].append(image_rel)
        label_path = label_by_split_stem.get(key)
        has_label = label_path is not None
        if not has_label:
            image_status = "missing_label"
            add_issue(issues, "error", "image_without_label", image_rel, "image has no matching label file", split=split)

        image_rows.append(
            {
                "split": split,
                "path": image_rel,
                "stem": stem,
                "scene_id": scene_id(stem),
                "extension": image_path.suffix.lower(),
                "width": width,
                "height": height,
                "mode": mode,
                "size_bytes": image_path.stat().st_size,
                "sha256": image_hash,
                "has_label": has_label,
                "status": image_status,
                "long_side_bucket": bucket_long_side(int(width), int(height)) if width and height else "",
            }
        )

    for key, label_path in label_by_split_stem.items():
        split, stem = key
        label_rel = rel(label_path, repo_root)
        image_path = image_by_split_stem.get(key)
        if image_path is None:
            add_issue(
                issues,
                "error",
                "label_without_image",
                label_rel,
                "label file has no matching image file",
                split=split,
            )

        width = height = None
        if image_path is not None:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception:
                pass

        text = label_path.read_text(encoding="utf-8", errors="replace")
        raw_lines = text.splitlines()
        nonempty_lines = [line for line in raw_lines if line.strip()]
        if not nonempty_lines:
            add_issue(issues, "warn", "empty_label_file", label_rel, "label file has no objects", split=split)

        valid_objects = 0
        for line_no, raw_line in enumerate(raw_lines, start=1):
            line = raw_line.strip()
            if not line:
                add_issue(issues, "warn", "blank_label_line", label_rel, "blank line in label file", split=split, line_no=line_no)
                continue
            parts = line.split()
            if len(parts) != 5:
                add_issue(
                    issues,
                    "error",
                    "invalid_yolo_column_count",
                    label_rel,
                    f"expected 5 columns, got {len(parts)}",
                    split=split,
                    line_no=line_no,
                    value=line,
                )
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, box_w, box_h = (float(v) for v in parts[1:])
            except ValueError:
                add_issue(
                    issues,
                    "error",
                    "invalid_yolo_number",
                    label_rel,
                    "class id or bbox coordinates are not valid numbers",
                    split=split,
                    line_no=line_no,
                    value=line,
                )
                continue

            object_status = "ok"
            values = [x_center, y_center, box_w, box_h]
            if class_id not in EXPECTED_CLASSES:
                object_status = "invalid_class_id"
                add_issue(
                    issues,
                    "error",
                    "invalid_class_id",
                    label_rel,
                    f"class id {class_id} is outside 0-24",
                    split=split,
                    line_no=line_no,
                    class_id=class_id,
                )
            if any(not math.isfinite(v) for v in values):
                object_status = "non_finite_coordinate"
                add_issue(
                    issues,
                    "error",
                    "non_finite_coordinate",
                    label_rel,
                    "bbox contains NaN or Inf",
                    split=split,
                    line_no=line_no,
                    class_id=class_id,
                    value=" ".join(parts[1:]),
                )
            if any(v < 0 or v > 1 for v in values):
                object_status = "coordinate_out_of_range"
                add_issue(
                    issues,
                    "error",
                    "coordinate_out_of_range",
                    label_rel,
                    "YOLO normalized coordinates must be within 0-1",
                    split=split,
                    line_no=line_no,
                    class_id=class_id,
                    value=" ".join(parts[1:]),
                )
            if box_w <= 0 or box_h <= 0:
                object_status = "non_positive_box"
                add_issue(
                    issues,
                    "error",
                    "non_positive_box",
                    label_rel,
                    "bbox width and height must be positive",
                    split=split,
                    line_no=line_no,
                    class_id=class_id,
                    value=f"{box_w} {box_h}",
                )

            x1 = x_center - box_w / 2
            y1 = y_center - box_h / 2
            x2 = x_center + box_w / 2
            y2 = y_center + box_h / 2
            if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
                if object_status == "ok":
                    object_status = "box_extends_outside_image"
                add_issue(
                    issues,
                    "warn",
                    "box_extends_outside_image",
                    label_rel,
                    "box edges exceed normalized image bounds after xywh to xyxy conversion",
                    split=split,
                    line_no=line_no,
                    class_id=class_id,
                    value=f"{x1:.8f} {y1:.8f} {x2:.8f} {y2:.8f}",
                )

            width_px = box_w * width if width else ""
            height_px = box_h * height if height else ""
            area_px = width_px * height_px if width and height else ""
            area_norm = box_w * box_h
            aspect_ratio = box_w / box_h if box_h > 0 else ""
            candidate_flags: list[str] = []
            if width and height and box_w > 0 and box_h > 0:
                if width_px < 4 or height_px < 4 or area_px < 16:
                    candidate_flags.append("tiny_box")
                if area_norm > 0.35 or box_w > 0.90 or box_h > 0.90:
                    candidate_flags.append("very_large_box")
                if aspect_ratio and (aspect_ratio > 15 or aspect_ratio < 1 / 15):
                    candidate_flags.append("extreme_aspect_ratio")
                if x1 <= 0.001 or y1 <= 0.001 or x2 >= 0.999 or y2 >= 0.999:
                    candidate_flags.append("edge_touching_box")

            if object_status == "ok":
                valid_objects += 1

            class_name, group = EXPECTED_CLASSES.get(class_id, ("", ""))
            object_rows.append(
                {
                    "split": split,
                    "image_path": rel(image_path, repo_root) if image_path else "",
                    "label_path": label_rel,
                    "stem": stem,
                    "scene_id": scene_id(stem),
                    "line_no": line_no,
                    "class_id": class_id,
                    "class_name": class_name,
                    "group": group,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width_norm": box_w,
                    "height_norm": box_h,
                    "x1_norm": x1,
                    "y1_norm": y1,
                    "x2_norm": x2,
                    "y2_norm": y2,
                    "width_px": width_px,
                    "height_px": height_px,
                    "area_px": area_px,
                    "area_norm": area_norm,
                    "aspect_ratio": aspect_ratio,
                    "candidate_flags": ";".join(candidate_flags),
                    "status": object_status,
                }
            )

        label_file_rows.append(
            {
                "split": split,
                "path": label_rel,
                "stem": stem,
                "size_bytes": label_path.stat().st_size,
                "sha256": sha256_file(label_path),
                "line_count": len(raw_lines),
                "object_count": valid_objects,
                "status": "ok" if image_path is not None else "label_without_image",
            }
        )

    for image_hash, paths in sorted(image_hash_to_paths.items()):
        if len(paths) > 1:
            for path in paths:
                duplicate_rows.append(
                    {
                        "sha256": image_hash,
                        "duplicate_count": len(paths),
                        "path": path,
                    }
                )

    image_keys = set(image_by_split_stem)
    label_keys = set(label_by_split_stem)
    for split, stem in sorted(image_keys - label_keys):
        add_issue(
            issues,
            "error",
            "image_without_label",
            rel(image_by_split_stem[(split, stem)], repo_root),
            "image has no matching label file",
            split=split,
        )
    for split, stem in sorted(label_keys - image_keys):
        add_issue(
            issues,
            "error",
            "label_without_image",
            rel(label_by_split_stem[(split, stem)], repo_root),
            "label file has no matching image file",
            split=split,
        )

    class_counter = Counter()
    split_class_counter = Counter()
    group_counter = Counter()
    split_group_counter = Counter()
    for row in object_rows:
        if row["status"] in {"invalid_class_id", "invalid_yolo_number", "invalid_yolo_column_count"}:
            continue
        class_counter[int(row["class_id"])] += 1
        split_class_counter[(row["split"], int(row["class_id"]))] += 1
        group_counter[row["group"]] += 1
        split_group_counter[(row["split"], row["group"])] += 1

    class_rows = []
    for class_id, (name, group) in EXPECTED_CLASSES.items():
        train_count = split_class_counter.get(("train", class_id), 0)
        val_count = split_class_counter.get(("val", class_id), 0)
        total = class_counter.get(class_id, 0)
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "group": group,
                "train_objects": train_count,
                "val_objects": val_count,
                "total_objects": total,
                "val_ratio": round(val_count / total, 6) if total else "",
            }
        )

    split_rows = []
    splits = sorted({row["split"] for row in image_rows} | {row["split"] for row in label_file_rows})
    for split in splits:
        split_images = [row for row in image_rows if row["split"] == split]
        split_labels = [row for row in label_file_rows if row["split"] == split and row["status"] != "ignored_auxiliary_file"]
        split_objects = [row for row in object_rows if row["split"] == split]
        split_rows.append(
            {
                "split": split,
                "images": len(split_images),
                "label_files": len(split_labels),
                "objects": len(split_objects),
                "ship_objects": split_group_counter.get((split, "ship"), 0),
                "aircraft_objects": split_group_counter.get((split, "aircraft"), 0),
                "vehicle_objects": split_group_counter.get((split, "vehicle"), 0),
            }
        )

    size_counter = Counter((row["split"], row["long_side_bucket"]) for row in image_rows)
    size_rows = [
        {"split": split, "long_side_bucket": bucket, "image_count": count}
        for (split, bucket), count in sorted(size_counter.items())
    ]

    scene_to_splits: dict[str, set[str]] = defaultdict(set)
    scene_to_images: dict[str, int] = Counter()
    for row in image_rows:
        scene_to_splits[row["scene_id"]].add(row["split"])
        scene_to_images[row["scene_id"]] += 1

    scene_rows = [
        {
            "scene_id": sid,
            "image_count": scene_to_images[sid],
            "splits": ";".join(sorted(scene_to_splits[sid])),
        }
        for sid in sorted(scene_to_images)
    ]
    scene_overlap_rows = [row for row in scene_rows if ";" in row["splits"]]
    for row in scene_overlap_rows:
        add_issue(
            issues,
            "warn",
            "scene_group_split_overlap",
            row["scene_id"],
            "same inferred scene id appears in multiple splits; random crop split may leak scene context",
            value=row["splits"],
        )

    issue_counter = Counter((row["severity"], row["issue_type"]) for row in issues)
    issue_summary_rows = [
        {"severity": severity, "issue_type": issue_type, "count": count}
        for (severity, issue_type), count in sorted(issue_counter.items())
    ]

    manifest_digest = hashlib.sha256()
    for row in sorted(image_rows, key=lambda item: item["path"]):
        manifest_digest.update(f"image\t{row['path']}\t{row['sha256']}\n".encode("utf-8"))
    for row in sorted(label_file_rows, key=lambda item: item["path"]):
        manifest_digest.update(f"label\t{row['path']}\t{row['sha256']}\n".encode("utf-8"))
    dataset_fingerprint = manifest_digest.hexdigest()

    fieldsets = {
        "image_manifest.csv": [
            "split",
            "path",
            "stem",
            "scene_id",
            "extension",
            "width",
            "height",
            "mode",
            "size_bytes",
            "sha256",
            "has_label",
            "status",
            "long_side_bucket",
        ],
        "label_manifest.csv": [
            "split",
            "path",
            "stem",
            "size_bytes",
            "sha256",
            "line_count",
            "object_count",
            "status",
        ],
        "objects.csv": [
            "split",
            "image_path",
            "label_path",
            "stem",
            "scene_id",
            "line_no",
            "class_id",
            "class_name",
            "group",
            "x_center",
            "y_center",
            "width_norm",
            "height_norm",
            "x1_norm",
            "y1_norm",
            "x2_norm",
            "y2_norm",
            "width_px",
            "height_px",
            "area_px",
            "area_norm",
            "aspect_ratio",
            "candidate_flags",
            "status",
        ],
        "label_issues.csv": ["severity", "issue_type", "split", "path", "line_no", "class_id", "value", "message"],
        "class_distribution.csv": ["class_id", "class_name", "group", "train_objects", "val_objects", "total_objects", "val_ratio"],
        "split_summary.csv": ["split", "images", "label_files", "objects", "ship_objects", "aircraft_objects", "vehicle_objects"],
        "image_size_buckets.csv": ["split", "long_side_bucket", "image_count"],
        "duplicate_images.csv": ["sha256", "duplicate_count", "path"],
        "scene_groups.csv": ["scene_id", "image_count", "splits"],
        "scene_split_overlap.csv": ["scene_id", "image_count", "splits"],
        "issue_summary.csv": ["severity", "issue_type", "count"],
        "class_contract_check.csv": ["class_id", "expected_name", "yaml_name", "group", "status"],
    }

    write_csv(manifest_dir / "image_manifest.csv", image_rows, fieldsets["image_manifest.csv"])
    write_csv(manifest_dir / "label_manifest.csv", label_file_rows, fieldsets["label_manifest.csv"])
    write_csv(output_dir / "objects.csv", object_rows, fieldsets["objects.csv"])
    write_csv(output_dir / "label_issues.csv", issues, fieldsets["label_issues.csv"])
    write_csv(output_dir / "class_distribution.csv", class_rows, fieldsets["class_distribution.csv"])
    write_csv(output_dir / "split_summary.csv", split_rows, fieldsets["split_summary.csv"])
    write_csv(output_dir / "image_size_buckets.csv", size_rows, fieldsets["image_size_buckets.csv"])
    write_csv(output_dir / "duplicate_images.csv", duplicate_rows, fieldsets["duplicate_images.csv"])
    write_csv(output_dir / "scene_groups.csv", scene_rows, fieldsets["scene_groups.csv"])
    write_csv(output_dir / "scene_split_overlap.csv", scene_overlap_rows, fieldsets["scene_split_overlap.csv"])
    write_csv(output_dir / "issue_summary.csv", issue_summary_rows, fieldsets["issue_summary.csv"])
    write_csv(output_dir / "class_contract_check.csv", class_contract_rows, fieldsets["class_contract_check.csv"])

    summary = {
        "audit_version": args.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": rel(data_root, repo_root),
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "image_count": len(image_rows),
        "label_file_count": len([row for row in label_file_rows if row["status"] != "ignored_auxiliary_file"]),
        "ignored_auxiliary_label_files": len(ignored_label_paths),
        "object_count": len(object_rows),
        "class_count": len(EXPECTED_CLASSES),
        "duplicate_image_rows": len(duplicate_rows),
        "scene_group_count": len(scene_rows),
        "scene_group_split_overlap_count": len(scene_overlap_rows),
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
        "warn_count": sum(1 for issue in issues if issue["severity"] == "warn"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (fingerprint_dir / f"{args.version}_dataset_fingerprint.txt").write_text(
        dataset_fingerprint + "\n", encoding="utf-8"
    )

    least_classes = sorted(class_rows, key=lambda row: row["total_objects"])[:10]
    most_classes = sorted(class_rows, key=lambda row: row["total_objects"], reverse=True)[:10]

    report = f"""# 数据审计报告：{args.version}

生成时间：{summary["generated_at_utc"]}

数据根目录：`{summary["data_root"]}`

数据指纹：`{dataset_fingerprint}`

## 结论摘要

- 图片数：{summary["image_count"]}
- 标注文件数：{summary["label_file_count"]}
- 忽略的辅助标签文件数：{summary["ignored_auxiliary_label_files"]}
- 标注目标数：{summary["object_count"]}
- 类别数：{summary["class_count"]}
- 精确重复图片记录数：{summary["duplicate_image_rows"]}
- 推断场景组数：{summary["scene_group_count"]}
- 跨 split 场景组数：{summary["scene_group_split_overlap_count"]}
- 问题记录数：{summary["issue_count"]}，其中 error {summary["error_count"]}，warn {summary["warn_count"]}

## split 摘要

{markdown_table(split_rows, fieldsets["split_summary.csv"])}

## 类别分布

{markdown_table(class_rows, fieldsets["class_distribution.csv"])}

## 数量最少的类别

{markdown_table(least_classes, ["class_id", "class_name", "group", "total_objects", "train_objects", "val_objects"])}

## 数量最多的类别

{markdown_table(most_classes, ["class_id", "class_name", "group", "total_objects", "train_objects", "val_objects"])}

## 图片尺寸分布

{markdown_table(size_rows, fieldsets["image_size_buckets.csv"])}

## 问题类型汇总

{markdown_table(issue_summary_rows, fieldsets["issue_summary.csv"])}

## 审计输出文件

- `shiyan/data_registry/manifests/{args.version}/image_manifest.csv`
- `shiyan/data_registry/manifests/{args.version}/label_manifest.csv`
- `shiyan/data_registry/audits/{args.version}/objects.csv`
- `shiyan/data_registry/audits/{args.version}/class_distribution.csv`
- `shiyan/data_registry/audits/{args.version}/split_summary.csv`
- `shiyan/data_registry/audits/{args.version}/image_size_buckets.csv`
- `shiyan/data_registry/audits/{args.version}/duplicate_images.csv`
- `shiyan/data_registry/audits/{args.version}/scene_groups.csv`
- `shiyan/data_registry/audits/{args.version}/scene_split_overlap.csv`
- `shiyan/data_registry/audits/{args.version}/label_issues.csv`
- `shiyan/data_registry/audits/{args.version}/issue_summary.csv`
- `shiyan/data_registry/audits/{args.version}/summary.json`
- `shiyan/data_registry/fingerprints/{args.version}_dataset_fingerprint.txt`

## 后续处理原则

本次审计只读取原始数据，不修改 `shiyan/data/`。若后续人工修标，应先在问题表中登记，再生成新的标签版本，例如 `labels_v1_cleaned`，并对新版本重新审计。
"""
    (output_dir / "audit_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the original competition YOLO dataset.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--data-root", default="shiyan/data", help="Dataset root containing images/, labels/, dataset.yaml.")
    parser.add_argument("--output-root", default="shiyan/data_registry", help="Output root for audit reports.")
    parser.add_argument("--version", default="v0_original", help="Audit version name.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(audit(parse_args()))
