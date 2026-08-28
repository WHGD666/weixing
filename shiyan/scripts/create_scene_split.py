#!/usr/bin/env python
"""Create a reproducible scene-grouped train/val split from audit outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def score_candidate(
    val_scene_ids: set[str],
    scenes: dict[str, dict[str, Any]],
    total_images: int,
    total_objects: int,
    class_totals: Counter[int],
    target_val_ratio: float,
) -> tuple[float, dict[str, Any]]:
    val_images = sum(scenes[sid]["image_count"] for sid in val_scene_ids)
    val_objects = sum(scenes[sid]["object_count"] for sid in val_scene_ids)
    val_class_counts: Counter[int] = Counter()
    for sid in val_scene_ids:
        val_class_counts.update(scenes[sid]["class_counts"])

    image_ratio = val_images / total_images
    object_ratio = val_objects / total_objects if total_objects else 0.0
    score = abs(image_ratio - target_val_ratio) * 8.0
    score += abs(object_ratio - target_val_ratio) * 4.0

    missing_val_classes: list[int] = []
    missing_train_classes: list[int] = []
    rare_underfilled_classes: list[int] = []
    class_ratio_errors = []

    for class_id, total in sorted(class_totals.items()):
        val_count = val_class_counts[class_id]
        train_count = total - val_count
        val_ratio = val_count / total if total else 0.0
        weight = 1.0
        if total < 100:
            weight = 6.0
        elif total < 500:
            weight = 3.0
        elif total < 1000:
            weight = 1.5
        ratio_error = abs(val_ratio - target_val_ratio)
        class_ratio_errors.append(ratio_error)
        score += ratio_error * weight

        if val_count == 0:
            missing_val_classes.append(class_id)
            score += 200.0
        if train_count == 0:
            missing_train_classes.append(class_id)
            score += 200.0
        if total < 100:
            min_expected = max(1, round(total * target_val_ratio * 0.5))
            if val_count < min_expected:
                rare_underfilled_classes.append(class_id)
                score += (min_expected - val_count) * 20.0

    diagnostics = {
        "val_images": val_images,
        "train_images": total_images - val_images,
        "val_objects": val_objects,
        "train_objects": total_objects - val_objects,
        "image_ratio": image_ratio,
        "object_ratio": object_ratio,
        "mean_class_ratio_error": sum(class_ratio_errors) / len(class_ratio_errors),
        "max_class_ratio_error": max(class_ratio_errors),
        "missing_val_classes": missing_val_classes,
        "missing_train_classes": missing_train_classes,
        "rare_underfilled_classes": rare_underfilled_classes,
    }
    return score, diagnostics


def choose_split(
    scenes: dict[str, dict[str, Any]],
    target_val_ratio: float,
    seed: int,
    trials: int,
) -> tuple[set[str], dict[str, Any]]:
    scene_ids = sorted(scenes)
    total_images = sum(scene["image_count"] for scene in scenes.values())
    total_objects = sum(scene["object_count"] for scene in scenes.values())
    class_totals: Counter[int] = Counter()
    for scene in scenes.values():
        class_totals.update(scene["class_counts"])

    target_val_images = round(total_images * target_val_ratio)
    best_score = float("inf")
    best_val_scene_ids: set[str] = set()
    best_diagnostics: dict[str, Any] = {}

    for trial in range(trials):
        rng = random.Random(seed + trial)
        shuffled = scene_ids[:]
        rng.shuffle(shuffled)

        val_scene_ids: set[str] = set()
        val_images = 0
        for sid in shuffled:
            scene_images = scenes[sid]["image_count"]
            if val_images < target_val_images:
                val_scene_ids.add(sid)
                val_images += scene_images
            else:
                break

        score, diagnostics = score_candidate(
            val_scene_ids,
            scenes,
            total_images,
            total_objects,
            class_totals,
            target_val_ratio,
        )
        if score < best_score:
            best_score = score
            best_val_scene_ids = val_scene_ids
            best_diagnostics = diagnostics | {"score": best_score, "trial": trial, "seed_used": seed + trial}

    return best_val_scene_ids, best_diagnostics


def split_fingerprint(assignments: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(assignments, key=lambda item: item["image_path"]):
        digest.update(f"{row['image_path']}\t{row['assigned_split']}\t{row['scene_id']}\n".encode("utf-8"))
    return digest.hexdigest()


def create_dataset_yaml(path: Path, split_version: str) -> None:
    names = {class_id: name for class_id, (name, _group) in EXPECTED_CLASSES.items()}
    data = {
        "path": ".",
        "train": f"shiyan/data_registry/split_assignments/{split_version}/train.txt",
        "val": f"shiyan/data_registry/split_assignments/{split_version}/val.txt",
        "nc": len(EXPECTED_CLASSES),
        "names": names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a scene-grouped 80/20 train/val split.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--audit-version", default="v0_original", help="Input audit version.")
    parser.add_argument("--split-version", default="v1_scene_80_20", help="Output split version.")
    parser.add_argument("--target-val-ratio", type=float, default=0.20, help="Target validation image ratio.")
    parser.add_argument("--seed", type=int, default=20260828, help="Base random seed.")
    parser.add_argument("--trials", type=int, default=5000, help="Random candidate search trials.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_registry = repo_root / "shiyan" / "data_registry"
    image_manifest_path = data_registry / "manifests" / args.audit_version / "image_manifest.csv"
    objects_path = data_registry / "audits" / args.audit_version / "objects.csv"
    output_dir = data_registry / "split_assignments" / args.split_version
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rows = read_csv(image_manifest_path)
    object_rows = read_csv(objects_path)

    scenes: dict[str, dict[str, Any]] = {}
    images_by_path: dict[str, dict[str, str]] = {}
    for row in image_rows:
        if row["status"] != "ok":
            continue
        scene_id = row["scene_id"]
        scenes.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "image_count": 0,
                "object_count": 0,
                "class_counts": Counter(),
                "image_paths": [],
            },
        )
        scenes[scene_id]["image_count"] += 1
        scenes[scene_id]["image_paths"].append(row["path"])
        images_by_path[row["path"]] = row

    for row in object_rows:
        if row["status"] != "ok":
            continue
        scene_id = row["scene_id"]
        class_id = int(row["class_id"])
        scenes[scene_id]["object_count"] += 1
        scenes[scene_id]["class_counts"][class_id] += 1

    val_scene_ids, diagnostics = choose_split(
        scenes=scenes,
        target_val_ratio=args.target_val_ratio,
        seed=args.seed,
        trials=args.trials,
    )

    image_assignments: list[dict[str, Any]] = []
    for row in sorted(image_rows, key=lambda item: item["path"]):
        if row["status"] != "ok":
            continue
        assigned_split = "val" if row["scene_id"] in val_scene_ids else "train"
        image_assignments.append(
            {
                "assigned_split": assigned_split,
                "original_split": row["split"],
                "image_path": row["path"],
                "label_path": row["path"].replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt",
                "stem": row["stem"],
                "scene_id": row["scene_id"],
                "width": row["width"],
                "height": row["height"],
                "sha256": row["sha256"],
            }
        )

    fingerprint = split_fingerprint(image_assignments)

    train_paths = [row["image_path"] for row in image_assignments if row["assigned_split"] == "train"]
    val_paths = [row["image_path"] for row in image_assignments if row["assigned_split"] == "val"]
    (output_dir / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_paths) + "\n", encoding="utf-8")

    scene_rows = []
    for scene_id, scene in sorted(scenes.items()):
        assigned_split = "val" if scene_id in val_scene_ids else "train"
        scene_rows.append(
            {
                "scene_id": scene_id,
                "assigned_split": assigned_split,
                "image_count": scene["image_count"],
                "object_count": scene["object_count"],
                "class_counts_json": json.dumps(dict(sorted(scene["class_counts"].items())), ensure_ascii=False),
            }
        )

    class_counts_by_split: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    group_counts_by_split: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    image_split_lookup = {row["image_path"]: row["assigned_split"] for row in image_assignments}
    for row in object_rows:
        if row["status"] != "ok":
            continue
        assigned_split = image_split_lookup[row["image_path"]]
        class_id = int(row["class_id"])
        _name, group = EXPECTED_CLASSES[class_id]
        class_counts_by_split[assigned_split][class_id] += 1
        group_counts_by_split[assigned_split][group] += 1

    class_rows = []
    for class_id, (name, group) in EXPECTED_CLASSES.items():
        train_objects = class_counts_by_split["train"][class_id]
        val_objects = class_counts_by_split["val"][class_id]
        total = train_objects + val_objects
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "group": group,
                "train_objects": train_objects,
                "val_objects": val_objects,
                "total_objects": total,
                "val_ratio": round(val_objects / total, 6) if total else "",
            }
        )

    group_rows = []
    for group in ["ship", "aircraft", "vehicle"]:
        train_objects = group_counts_by_split["train"][group]
        val_objects = group_counts_by_split["val"][group]
        total = train_objects + val_objects
        group_rows.append(
            {
                "group": group,
                "train_objects": train_objects,
                "val_objects": val_objects,
                "total_objects": total,
                "val_ratio": round(val_objects / total, 6) if total else "",
            }
        )

    split_summary_rows = []
    for split in ["train", "val"]:
        split_images = [row for row in image_assignments if row["assigned_split"] == split]
        split_scenes = [row for row in scene_rows if row["assigned_split"] == split]
        split_summary_rows.append(
            {
                "split": split,
                "images": len(split_images),
                "image_ratio": round(len(split_images) / len(image_assignments), 6),
                "scenes": len(split_scenes),
                "objects": sum(class_counts_by_split[split].values()),
                "object_ratio": round(sum(class_counts_by_split[split].values()) / len(object_rows), 6),
            }
        )

    train_scenes = {row["scene_id"] for row in scene_rows if row["assigned_split"] == "train"}
    val_scenes = {row["scene_id"] for row in scene_rows if row["assigned_split"] == "val"}
    overlap_scenes = sorted(train_scenes & val_scenes)
    missing_val_classes = [row for row in class_rows if int(row["val_objects"]) == 0]
    missing_train_classes = [row for row in class_rows if int(row["train_objects"]) == 0]

    audit_checks = [
        {
            "check": "scene_disjointness",
            "status": "pass" if not overlap_scenes else "fail",
            "detail": f"{len(overlap_scenes)} overlapping scenes",
        },
        {
            "check": "all_classes_present_in_train",
            "status": "pass" if not missing_train_classes else "fail",
            "detail": ",".join(str(row["class_id"]) for row in missing_train_classes),
        },
        {
            "check": "all_classes_present_in_val",
            "status": "pass" if not missing_val_classes else "fail",
            "detail": ",".join(str(row["class_id"]) for row in missing_val_classes),
        },
        {
            "check": "target_val_image_ratio",
            "status": "pass"
            if abs(split_summary_rows[1]["image_ratio"] - args.target_val_ratio) <= 0.01
            else "warn",
            "detail": str(split_summary_rows[1]["image_ratio"]),
        },
    ]

    metadata = {
        "split_version": args.split_version,
        "audit_version": args.audit_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "trials": args.trials,
        "target_val_ratio": args.target_val_ratio,
        "split_unit": "scene_id inferred from filename before _cropN",
        "original_data_policy": "No files were moved, copied, renamed, or modified.",
        "split_fingerprint_sha256": fingerprint,
        "diagnostics": diagnostics,
    }

    write_csv(
        output_dir / "image_assignments.csv",
        image_assignments,
        ["assigned_split", "original_split", "image_path", "label_path", "stem", "scene_id", "width", "height", "sha256"],
    )
    write_csv(
        output_dir / "scene_assignments.csv",
        scene_rows,
        ["scene_id", "assigned_split", "image_count", "object_count", "class_counts_json"],
    )
    write_csv(
        output_dir / "class_distribution_by_split.csv",
        class_rows,
        ["class_id", "class_name", "group", "train_objects", "val_objects", "total_objects", "val_ratio"],
    )
    write_csv(
        output_dir / "group_distribution_by_split.csv",
        group_rows,
        ["group", "train_objects", "val_objects", "total_objects", "val_ratio"],
    )
    write_csv(
        output_dir / "split_summary.csv",
        split_summary_rows,
        ["split", "images", "image_ratio", "scenes", "objects", "object_ratio"],
    )
    write_csv(output_dir / "audit_checks.csv", audit_checks, ["check", "status", "detail"])
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "split_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")

    dataset_yaml_path = repo_root / "shiyan" / "configs" / "dataset" / f"weixing_{args.split_version}.yaml"
    create_dataset_yaml(dataset_yaml_path, args.split_version)

    rare_rows = sorted(class_rows, key=lambda row: int(row["total_objects"]))[:8]
    report = f"""# 划分审计报告：{args.split_version}

生成时间：{metadata["generated_at_utc"]}

输入审计版本：`{args.audit_version}`

划分指纹：`{fingerprint}`

## 划分原则

- 原始数据不移动、不复制、不重命名、不修改。
- 划分单元为 `scene_id`，即文件名中 `_cropN` 之前的部分。
- 同一个 `scene_id` 下的切片全部进入同一个 split。
- 目标比例为 train:val = 8:2。
- 在候选随机划分中选择类别比例更接近 8:2、且稀有类别验证集不为空的方案。

## split 摘要

{markdown_table(split_summary_rows, ["split", "images", "image_ratio", "scenes", "objects", "object_ratio"])}

## 大类分布

{markdown_table(group_rows, ["group", "train_objects", "val_objects", "total_objects", "val_ratio"])}

## 25 类分布

{markdown_table(class_rows, ["class_id", "class_name", "group", "train_objects", "val_objects", "total_objects", "val_ratio"])}

## 稀有类别覆盖

{markdown_table(rare_rows, ["class_id", "class_name", "group", "train_objects", "val_objects", "total_objects", "val_ratio"])}

## 自动检查

{markdown_table(audit_checks, ["check", "status", "detail"])}

## 输出文件

- `train.txt`：训练图片清单。
- `val.txt`：验证图片清单。
- `image_assignments.csv`：每张图片的 split 分配。
- `scene_assignments.csv`：每个场景组的 split 分配。
- `class_distribution_by_split.csv`：25 类 train/val 目标数量。
- `group_distribution_by_split.csv`：舰船、飞机、发射车三大类目标数量。
- `split_summary.csv`：图片、场景、目标数量摘要。
- `audit_checks.csv`：划分质量自动检查。
- `metadata.json`：划分参数、随机种子、搜索诊断和指纹。
- `split_fingerprint.txt`：划分指纹。
- `shiyan/configs/dataset/weixing_{args.split_version}.yaml`：YOLO 数据配置。

## 后续使用

训练时应优先使用 `shiyan/configs/dataset/weixing_{args.split_version}.yaml`，不要直接使用官方 `shiyan/data/dataset.yaml`。官方原始数据目录保持不变，后续所有正式实验都需要记录本 split 版本和划分指纹。
"""
    (output_dir / "split_audit_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(metadata | {"summary": split_summary_rows, "audit_checks": audit_checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
