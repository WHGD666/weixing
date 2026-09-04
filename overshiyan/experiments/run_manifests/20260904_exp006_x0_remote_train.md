# Run Manifest: 20260904 EXP006-X0 Remote Training

## Identity

| Field | Value |
| --- | --- |
| Run ID | `20260904_train_exp006_x0_5090` |
| Experiment | `EXP006_X0_data3_yolo11x_1024` |
| Role | competition model-capacity control |
| Status | running; no checkpoint selected |
| Started | approximately `2026-09-04 19:56 CST` |
| Training host | NVIDIA GeForce RTX 5090 32 GB |
| Deployment target | NVIDIA RTX 3090 24 GB |

## Frozen inputs

| Evidence | Value |
| --- | --- |
| Remote bundle | `overshiyan_remote_20260904.tar` |
| Bundle bytes | `1350321152` |
| Bundle SHA256 | `3476525e3cbc7fc4f9e6d9acef607e9ccfe87ea03718faeaadbd9696f411784f` |
| Data3 fingerprint | `7615f507bebade10696c4f32a708951e91133ee5736e08da9b4e7834a3e3b956` |
| Train/val split SHA256 | `a294781c04f122b7f75e545cf901171127b735b22f41c1d1974b9e58b5adbc68` / `a71aaef1231a72648b4050eb3c72a3c679603aff5644d2d03a665af588645d7f` |
| Pretrained model SHA256 | `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` |
| Train config SHA256 | `6977438db26e7e4e49cb4e343e74d7ffb9a24a5e45529fe29f2f5dfe7fbb64da` |
| Train entry SHA256 | `b99f0c8cdbc4583c6daf16dc09f2edaa71756ab69fe31c0b3a73566b0f23dd14` |
| Requirements SHA256 | `d24e6e829625bcc5a8fea1f9815a1d39451200e8b6ffb517b2ccb3d51610a134` |

The bundle was prepared from a dirty local `main` worktree whose last committed
parent was `f3e4d14`. The archive SHA256 and the hashes above, rather than the
parent commit alone, identify the exact files used by this already-running job.
The source code is being committed after launch so that subsequent evaluation
and reporting can reference a clean Git revision without rewriting run history.

## Data preflight

- Source: 4481 images, 4481 labels, 25 classes, 21008 objects.
- Intentional empty labels: `0581`, `0582`, `0583`, `0910`.
- Invalid or unreadable pairs: none.
- Prepared split: 3584 train images / 897 validation images, zero overlap.
- Prepared objects: 16756 train / 4250 validation.
- Two exact duplicate rows were removed from the generated view only; Data3
  remained read-only.
- Ultralytics loader audit and fixed EXP006 preflight both returned `ok=True`.

## Fixed command and configuration

```bash
nohup python -u scripts/03_train_exp006.py \
  > runs/train/EXP006_console.log 2>&1 &
```

The fixed schedule is YOLO11x, 1024 px, batch 8, 60 epochs, AMP, deterministic
seed 614651, checkpoint every five epochs, and a 3.75-hour epoch-boundary guard.
EXP006 does not include rare-class resampling, copy-paste, hard-negative mining,
ensembling, modality routing, or 1280 fine-tuning.

## Transfer incident

The initial web-panel transfer failed SHA256 verification and tar integrity with
`Unexpected EOF`; its partial extraction contained 2922 images and was rejected.
The second SCP transfer matched the expected byte count and SHA256 and passed
`tar -tf` before extraction. No training used the damaged package.

## Interim health evidence

| Checkpoint | P | R | mAP50 | mAP50-95 | Projected 60 epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch 2 | 0.405 | 0.532 | 0.471 | 0.238 | 3.50 h |
| epoch 6 | 0.732 | 0.709 | 0.732 | 0.416 | 2.40 h |

Observed GPU state: 97% utilization, approximately 20.5 GB used VRAM, 63 C, and
529 W. These aggregate Ultralytics values only show that optimization is healthy.
They are not D3/D0 three-group Recall/FDR and are not official hidden-test scores.

## Required closeout

1. Preserve `results.csv`, `args.yaml`, `epoch_timings.csv`, `run_manifest.json`,
   `best.pt`, `last.pt`, and useful periodic checkpoints.
2. Hash every evaluated checkpoint.
3. Run full 897-image inference once per candidate and score identical predictions
   against D3 and mapped D0 labels.
4. Rank by worst-protocol gate margin, not Ultralytics fitness alone.
5. Freeze no model until prediction schema, RTX 3090 timing, and Docker offline
   execution pass.
6. Record any formal platform result separately from this internal evidence.
