# Overshiyan: Data3 + YOLO11x

This directory is an isolated, reproducible training and validation workspace for
the final competition phase. The source dataset under `data3/` is treated as
read-only. Every generated split, cleaned label, model, metric, and submission
artifact is written elsewhere and recorded with hashes.

Current status on 2026-09-04: `EXP006-X0` passed source, loader, environment, and
fixed-configuration preflight and is running on the RTX 5090 host. Epoch-level
Ultralytics values remain interim evidence; no checkpoint or submission candidate
has been selected. See
[`experiments/run_manifests/20260904_exp006_x0_remote_train.md`](experiments/run_manifests/20260904_exp006_x0_remote_train.md).

## Fixed experimental contract

- Training host: RTX 5090 32 GB, PyTorch 2.10, Python 3.12, CUDA 12.8.
- Deployment target: RTX 3090 24 GB, CUDA 12.1-compatible container.
- Dataset: manually revised Data3, 4481 images, 25 exact classes.
- Split: fixed scene-level 3584 train / 897 validation assignment.
- Main control: `EXP006-X0`, official `yolo11x.pt`, image size 1024, 60 epochs.
- Validation: the same predictions are scored against Data3 labels and the
  mapped original-label D0 protocol. Selection uses the worse protocol result.
- Hard gates: three-group macro Recall >= 0.85, FDR <= 0.20, and image latency
  <= 20 seconds. Local safety targets are Recall >= 0.90 and FDR <= 0.12.
- Validation images and labels are never augmented or used for hard-negative
  mining.
- Boundary-crossing boxes are retained and reported. They are expected evidence
  of the organizer's large-image tiling policy, not silently treated as corrupt.

## Directory contract

```text
data3/                         immutable source dataset
configs/                       reviewed train/inference/deploy settings
data_registry/                 class contract, split IDs, audits, protocols
models/pretrained/             place the official yolo11x.pt here
models/frozen/                 selected checkpoints and manifests
scripts/                       numbered operator entry points
src/                           shared implementation
workspace/                     generated train/val view (ignored)
runs/                          generated training/evaluation runs (ignored)
submit/                        RTX 3090 submission skeleton
```

## Remote-host sequence

Run these commands from the `overshiyan` root. They are deliberately separated
so a failed preflight cannot silently start a costly training job.

```bash
python -m pip install -r requirements-train.txt
python scripts/00_fetch_yolo11x.py
python scripts/00_env_check.py --target train5090
python scripts/01_audit_data3.py --strict
python scripts/02_prepare_data3_view.py
python scripts/02_validate_ultralytics_dataset.py
python scripts/03_train_exp006.py --preflight-only
python scripts/03_train_exp006.py
```

Do not start `EXP007-X1` before `EXP006-X0` has been evaluated. The rare-class
configuration is included as a reviewed next-stage skeleton, not as permission
to combine several unmeasured changes at once.

## After training

Inference, dual-protocol evaluation, checkpoint ranking, and submission freezing
are documented in [RUNBOOK.md](RUNBOOK.md). GPU-heavy commands remain operator
actions; the preparation and audit scripts are safe CPU tasks.

For the final-day transfer sequence, use [REMOTE_QUICKSTART.md](REMOTE_QUICKSTART.md).
Never send or commit an SSH private key, password, or registry token.
