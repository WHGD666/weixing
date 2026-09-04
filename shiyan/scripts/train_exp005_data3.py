#!/usr/bin/env python
"""Preflight and train EXP005 with delayed early stopping."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "shiyan/configs/train/exp005_data3_manual_revision_classfix.yaml"

# ============================================================
# EXP005 training strategy
#
# Maximum epochs: 150
# Epoch 1-80: Early stopping disabled
# Epoch 81+: Early stopping enabled
# Stop if fitness does not improve for 30 epochs
# ============================================================

MAX_EPOCHS = 150
EARLY_STOP_START_EPOCH = 80
EARLY_STOP_PATIENCE = 30

EXPECTED_SPLIT_COUNTS = {
    "train": 3584,
    "val": 897,
}

EXPECTED_EMPTY_LABELS = {
    "0581.txt",
    "0582.txt",
    "0583.txt",
    "0910.txt",
}

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


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path to an absolute path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_split(path: Path) -> list[str]:
    """Read train/val split file."""
    return [
        line.strip().replace("\\", "/")
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]


def preflight(config_path: Path) -> dict[str, str]:
    """
    Run all EXP005 preflight checks before training.

    This verifies:
    - training configuration
    - initialization model
    - dataset configuration
    - class order
    - YOLO labels
    - frozen train/val split
    - train/val overlap
    - empty labels
    - dataset fingerprint
    - SHA256 fingerprints
    """

    # --------------------------------------------------------
    # 1. Training config
    # --------------------------------------------------------

    if not config_path.is_file():
        raise SystemExit(
            f"Missing training config: {config_path}"
        )

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    ) or {}

    # EXP005 now uses at most 150 epochs.
    # patience=0 disables Ultralytics' built-in early stopping.
    # Our custom delayed early-stopping callback controls it.
    if (
        config.get("epochs") != MAX_EPOCHS
        or config.get("patience") != 0
    ):
        raise SystemExit(
            f"EXP005 config must use "
            f"epochs={MAX_EPOCHS} and patience=0; "
            f"delayed stopping is controlled by this launcher"
        )

    # --------------------------------------------------------
    # 2. Model + dataset configuration
    # --------------------------------------------------------

    model_path = resolve_repo_path(config["model"])
    dataset_config_path = resolve_repo_path(config["data"])

    if not model_path.is_file():
        raise SystemExit(
            f"Missing initialization model: {model_path}"
        )

    if not dataset_config_path.is_file():
        raise SystemExit(
            f"Missing dataset config: {dataset_config_path}"
        )

    # --------------------------------------------------------
    # 3. Dataset class contract
    # --------------------------------------------------------

    dataset_config = yaml.safe_load(
        dataset_config_path.read_text(encoding="utf-8")
    ) or {}

    names = dataset_config.get("names", {})

    actual_classes = [
        str(names.get(index, ""))
        for index in range(25)
    ]

    if (
        dataset_config.get("nc") != 25
        or actual_classes != EXPECTED_CLASSES
    ):
        raise SystemExit(
            "Dataset class contract mismatch; "
            "expected the fixed 25-class order"
        )

    # --------------------------------------------------------
    # 4. YOLO label checker
    # --------------------------------------------------------

    checker = (
        ROOT
        / "shiyan/scripts/check_data2_yolo_labels.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(checker),
            "--root",
            "shiyan/data3",
        ],
        cwd=ROOT,
        check=True,
    )

    # --------------------------------------------------------
    # 5. Frozen train/val split
    # --------------------------------------------------------

    split_sets: dict[str, set[str]] = {}

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():

        split_path = resolve_repo_path(
            dataset_config[split]
        )

        entries = read_split(split_path)

        # Check number of images
        if len(entries) != expected_count:
            raise SystemExit(
                f"{split} split count mismatch: "
                f"expected {expected_count}, "
                f"got {len(entries)}"
            )

        # Check duplicate entries
        if len(entries) != len(set(entries)):
            raise SystemExit(
                f"Duplicate image entries in {split} split"
            )

        # Check that every image exists
        missing = [
            entry
            for entry in entries
            if not resolve_repo_path(entry).is_file()
        ]

        if missing:
            raise SystemExit(
                f"Missing images in {split} split: "
                f"{missing[:5]}"
            )

        split_sets[split] = set(entries)

    # --------------------------------------------------------
    # 6. Train / validation leakage check
    # --------------------------------------------------------

    overlap = (
        split_sets["train"]
        & split_sets["val"]
    )

    if overlap:
        raise SystemExit(
            "Train/val overlap detected: "
            f"{sorted(overlap)[:5]}"
        )

    # Make sure all 4481 images are covered
    if len(
        split_sets["train"]
        | split_sets["val"]
    ) != 4481:
        raise SystemExit(
            "Frozen split does not cover "
            "all 4481 data3 images"
        )

    # --------------------------------------------------------
    # 7. Empty-label contract
    # --------------------------------------------------------

    label_root = (
        ROOT
        / "shiyan/data3/labels/train"
    )

    empty_labels = {
        path.name
        for path in label_root.glob("*.txt")
        if (
            path.name != "classes.txt"
            and path.stat().st_size == 0
        )
    }

    if empty_labels != EXPECTED_EMPTY_LABELS:
        raise SystemExit(
            "Unexpected empty-label set: "
            f"{sorted(empty_labels)}"
        )

    # --------------------------------------------------------
    # 8. Dataset fingerprint
    # --------------------------------------------------------

    fingerprint_path = (
        ROOT
        / "shiyan/data_registry/fingerprints/"
          "v2_data3_manual_revision_classfix_"
          "dataset_fingerprint.txt"
    )

    if not fingerprint_path.is_file():
        raise SystemExit(
            "Missing audited data fingerprint: "
            f"{fingerprint_path}"
        )

    data_fingerprint = (
        fingerprint_path
        .read_text(encoding="utf-8")
        .strip()
    )

    # --------------------------------------------------------
    # 9. Record reproducibility information
    # --------------------------------------------------------

    result = {
        "config":
            config_path.relative_to(ROOT).as_posix(),

        "config_sha256":
            sha256_file(config_path),

        "dataset_config_sha256":
            sha256_file(dataset_config_path),

        "initial_model":
            model_path.relative_to(ROOT).as_posix(),

        "initial_model_sha256":
            sha256_file(model_path),

        "data_fingerprint_sha256":
            data_fingerprint,

        "train_images":
            str(EXPECTED_SPLIT_COUNTS["train"]),

        "val_images":
            str(EXPECTED_SPLIT_COUNTS["val"]),

        "expected_output_dir":
            "runs/detect/runs/train/"
            "exp005_data3_manual_revision_classfix",

        "max_epochs":
            str(MAX_EPOCHS),

        # Epoch 1-80 disabled;
        # Early stopping is active from Epoch 81.
        "early_stop_start_epoch":
            str(EARLY_STOP_START_EPOCH + 1),

        "early_stop_patience":
            str(EARLY_STOP_PATIENCE),
    }

    print("preflight_ok=true")

    for key, value in result.items():
        print(f"{key}={value}")

    return result


def add_delayed_early_stopping(model: object) -> None:
    """
    Add delayed early stopping to Ultralytics YOLO.

    Strategy:
        Epoch 1-80:
            Early stopping disabled.

        Epoch 81+:
            Early stopping enabled.

        Patience:
            30 epochs.

        Maximum:
            150 epochs.
    """

    def on_train_start(trainer: object) -> None:
        """
        Configure early stopping when training starts
        or when training is resumed.
        """

        start_epoch = int(trainer.start_epoch)

        # If resuming from Epoch 80 or later,
        # immediately enable early stopping.
        if start_epoch >= EARLY_STOP_START_EPOCH:

            trainer.stopper.patience = (
                EARLY_STOP_PATIENCE
            )

            # Restart the early-stop counter from
            # the resumed training position.
            trainer.stopper.best_epoch = start_epoch

            trainer.stopper.best_fitness = float(
                trainer.best_fitness or 0.0
            )

            trainer.stopper.possible_stop = False

            print(
                "delayed_early_stopping="
                "enabled_after_resume "
                f"start_epoch={start_epoch + 1} "
                f"patience={EARLY_STOP_PATIENCE}"
            )

        else:
            # Prevent stopping during Epoch 1-80.
            trainer.stopper.patience = float("inf")

            print(
                "delayed_early_stopping="
                f"disabled_until_epoch_"
                f"{EARLY_STOP_START_EPOCH} "
                f"start_epoch={start_epoch + 1}"
            )

    def on_train_epoch_end(
        trainer: object
    ) -> None:
        """
        Enable Early Stopping after Epoch 80.

        trainer.epoch is zero-based:
            trainer.epoch == 79
            means Epoch 80 has completed.
        """

        completed_epoch = (
            int(trainer.epoch) + 1
        )

        # Exactly after Epoch 80 finishes:
        #
        # Epoch 81 begins with patience=30.
        if (
            completed_epoch
            == EARLY_STOP_START_EPOCH
        ):

            trainer.stopper.patience = (
                EARLY_STOP_PATIENCE
            )

            # Reset the baseline at Epoch 80.
            trainer.stopper.best_epoch = (
                completed_epoch
            )

            trainer.stopper.best_fitness = float(
                trainer.best_fitness or 0.0
            )

            trainer.stopper.possible_stop = False

            print(
                "delayed_early_stopping="
                "enabled "
                f"from_epoch="
                f"{completed_epoch + 1} "
                f"patience="
                f"{EARLY_STOP_PATIENCE}"
            )

    # Register Ultralytics callbacks
    model.add_callback(
        "on_train_start",
        on_train_start,
    )

    model.add_callback(
        "on_train_epoch_end",
        on_train_epoch_end,
    )


def main() -> None:
    """EXP005 training launcher."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="EXP005 training YAML config.",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run all preflight checks "
            "without starting training."
        ),
    )

    parser.add_argument(
        "--resume",
        help=(
            "Resume from an EXP005 "
            "last.pt checkpoint."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve config
    # --------------------------------------------------------

    config_path = resolve_repo_path(
        args.config
    ).resolve()

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------

    details = preflight(config_path)

    if args.preflight_only:
        return

    # Import Ultralytics only after preflight passes.
    from ultralytics import YOLO

    # --------------------------------------------------------
    # Resume training
    # --------------------------------------------------------

    if args.resume:

        resume_path = resolve_repo_path(
            args.resume
        ).resolve()

        if not resume_path.is_file():
            raise SystemExit(
                "Missing resume checkpoint: "
                f"{resume_path}"
            )

        model = YOLO(
            str(resume_path)
        )

        add_delayed_early_stopping(model)

        print(
            "training_mode=resume "
            f"checkpoint={resume_path}"
        )

        model.train(
            resume=str(resume_path)
        )

    # --------------------------------------------------------
    # Fresh training
    # --------------------------------------------------------

    else:

        model = YOLO(
            str(
                resolve_repo_path(
                    details["initial_model"]
                )
            )
        )

        add_delayed_early_stopping(model)

        print("training_mode=fresh")

        model.train(
            cfg=str(config_path)
        )


if __name__ == "__main__":
    main()