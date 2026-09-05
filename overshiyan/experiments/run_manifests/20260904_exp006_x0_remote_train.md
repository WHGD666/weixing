# Run Manifest: 20260904 EXP006-X0 Remote Training

## Identity

| Field | Value |
| --- | --- |
| Run ID | `20260904_train_exp006_x0_5090` |
| Experiment | `EXP006_X0_data3_yolo11x_1024` |
| Role | competition model-capacity control |
| Status | completed; `best.pt` selected for next RTX 3090 validation |
| Started | approximately `2026-09-04 19:56 CST` |
| Finished | approximately `2026-09-04 21:50 CST` |
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

## Completion evidence

The training process completed normally. `run_manifest.json` records
`status=COMPLETED`, start time `2026-09-04T11:56:52+00:00`, finish time
`2026-09-04T13:50:38+00:00`, and official YOLO11x initialization.

| Artifact | SHA256 | Bytes |
| --- | --- | ---: |
| `best.pt` | `111ae87e2911dcf35a68cb0b7d047e75f88ab62baa7367e690324133bbf2d67d` | 114496345 |
| `last.pt` | `af65a3b478c875fbe4fae882b01ed896de3c7ac2f3a6102de4e01a470422b1e2` | 114496345 |

Ultralytics selected epoch 55 as `best.pt`. Its aggregate validation metrics
were P/R/mAP50/mAP50-95 `0.93424 / 0.96302 / 0.96667 / 0.73339`. Epoch 60 ended
at `0.93984 / 0.95335 / 0.96289 / 0.72650`.

## Dual-protocol checkpoint evaluation

Both evaluated checkpoints used the same tiled inference contract:
`conf=0.10`, class thresholds `0.30` for `0-23` and `0.35` for `FSC`,
`imgsz=1024`, `tile_overlap=0.20`, `merge_iou=0.50`, `iou=0.60`,
`tile_batch=1`, and FP16. The inference configuration SHA256 is
`8158784300018b9923bd1cf456c6f2af2776f1fa05306fc68a85a0fd22ac87bc`.

| Candidate | Worst protocol Recall | Worst protocol FDR | D0 Recall/FDR | D3 Recall/FDR | Max image time | Decision |
| --- | ---: | ---: | --- | --- | ---: | --- |
| `best.pt` | 0.885259 | 0.185691 | 0.885259 / 0.185691 | 0.897675 / 0.176099 | 4.284 s | selected |
| `last.pt` | 0.886076 | 0.197365 | 0.886076 / 0.197365 | 0.894885 / 0.190603 | 4.344 s | rejected, FDR margin too thin |

`best.pt` passes the internal D0 and D3 rigid gates. It does not pass the stricter
planning safety target of Recall `>= 0.90` and FDR `<= 0.12`, so the next step is
submission engineering validation rather than an automatic formal push.

## Evidence archive

Local evidence was pulled under the ignored directory
`runs/remote_archive/EXP006_20260904/`:

- `weights/best.pt`, verified by SHA256;
- `weights/last.pt`, verified by SHA256;
- `eval/EXP006_best_full/`, including result, timings, manifest, and metrics;
- `eval/EXP006_last_full/`, including result, timings, manifest, and metrics;
- `EXP006_evidence_20260904.tar.gz`, containing training logs, `results.csv`,
  `args.yaml`, `epoch_timings.csv`, audits, configuration, plots, and manifest.

The evidence directory is not committed because it contains large/regenerable
run artifacts. The committed files retain the hashes, commands, metrics, and
decisions needed for reproducibility.

## Remaining closeout before any formal submission

1. Package `best.pt` only into the submission skeleton.
2. Run output-schema validation on the package result.
3. Run RTX 3090 24 GB smoke and full Docker validation with `tile_batch=1`.
4. Compare 3090 timing against the `20 s` per-image gate.
5. Record the package image digest before considering a formal platform tag.
6. Record any formal platform result separately from this internal evidence.
