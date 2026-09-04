# EXP006-X0: Data3 + YOLO11x at 1024

Status: **RUNNING on the RTX 5090 host; interim evidence only**

## Hypothesis

The YOLO11s detector underfits fine-grained ship classes. Increasing model
capacity while preserving the manually corrected Data3 labels should improve
ship/vehicle recall and reduce class-confusion false alarms without depending on
postprocessing tricks.

## Controlled factors

- Fixed Data3 scene split: 3584 train / 897 validation.
- Official `yolo11x.pt` initialization.
- Image size 1024 and stable remote environment.
- 60 epochs with a checkpoint every 5 epochs; this schedule is fixed before the
  run to fit the final-day compute budget.
- A 3.75-hour wall-clock guard ends at an epoch boundary and preserves validation
  and checkpoints if 60 epochs cannot finish inside the available time.
- Ordinary remote-sensing geometry/photometry augmentation only.
- No rare-class oversampling, offline copy-paste, hard-negative mining, ensemble,
  modality routing, or 1280 fine-tuning in this experiment.
- Validation is unchanged and scored against D3 and mapped D0 labels.

## Risks

- HM/LQS remain information-poor despite augmentation.
- Data3 corrections may differ from hidden-test annotation policy.
- The best Ultralytics fitness epoch may not be the best rigid-gate epoch.
- A 5090-safe inference batch may exceed 3090 memory; deployment is benchmarked
  independently with `tile_batch=1` as the safe default.

## Acceptance evidence

The candidate must pass source audit, prediction schema checks, dual-protocol
metrics, and an offline RTX 3090 Docker run. The preferred local safety target is
worst-protocol Recall >= 0.90 and FDR <= 0.12. The platform remains authoritative.

## 2026-09-04 remote launch record

The first provider-panel upload was rejected before training because
`sha256sum` failed and `tar` reported `Unexpected EOF`; its partial extraction
contained only 2922 of 4481 images. That directory was not used for EXP006.
The bundle was transferred again with SCP and accepted only after all three
checks passed:

- archive: `overshiyan_remote_20260904.tar`;
- bytes: `1350321152`;
- archive SHA256: `3476525e3cbc7fc4f9e6d9acef607e9ccfe87ea03718faeaadbd9696f411784f`;
- tar integrity: passed;
- extracted Data3 images: 4481;
- required fixed configuration: present.

The clean source audit then reported 4481 images, 4481 labels, 25 classes,
21008 source objects, the four intentional empty labels `0581`, `0582`, `0583`,
and `0910`, and no missing, invalid, or unreadable pairs. Its fingerprint is
`7615f507bebade10696c4f32a708951e91133ee5736e08da9b4e7834a3e3b956`.
The generated training view retained the frozen 3584/897 scene split and removed
exactly two duplicate label rows, producing 16756 train objects and 4250
validation objects.

Environment and identity checks passed before launch:

- Python 3.12.13;
- PyTorch 2.10.0+cu128;
- Ultralytics 8.4.128 (the update notice was intentionally ignored);
- NVIDIA GeForce RTX 5090, 32 GB;
- official YOLO11x, 56,966,176 parameters;
- pretrained SHA256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`;
- training configuration SHA256 `6977438db26e7e4e49cb4e343e74d7ffb9a24a5e45529fe29f2f5dfe7fbb64da`.

The host did not provide `tmux`, so the operator started one `nohup` process at
approximately 19:56 CST with PID 9605:

```bash
nohup python -u scripts/03_train_exp006.py \
  > runs/train/EXP006_console.log 2>&1 &
```

At the first hardware observation the process used about 20.5/32.6 GB VRAM,
97% GPU utilization, 63 C, and 529/600 W. At epoch 2 the Ultralytics aggregate
metrics were P/R/mAP50/mAP50-95 `0.405/0.532/0.471/0.238`; at epoch 6 they were
`0.732/0.709/0.732/0.416`. The projected 60-epoch duration fell from 3.50 hours
at epoch 2 to 2.40 hours at epoch 6.

These are training-health observations, not the competition metrics. They do not
establish the three-group Recall/FDR gates. Final decisions remain blocked until
best, last, and useful periodic checkpoints are evaluated on the complete 897
images under both D3 and mapped D0 protocols, followed by output-schema checks
and RTX 3090 timing/Docker validation.
