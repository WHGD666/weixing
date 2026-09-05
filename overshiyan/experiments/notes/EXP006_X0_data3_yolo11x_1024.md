# EXP006-X0: Data3 + YOLO11x at 1024

Status: **COMPLETED; best.pt is the selected EXP006 checkpoint for the next
submission-candidate validation step**

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
establish the three-group Recall/FDR gates.

## 2026-09-04 completion and checkpoint evidence

Training completed normally on the RTX 5090 host.

- Started: `2026-09-04T11:56:52+00:00` / `2026-09-04 19:56:52 CST`.
- Finished: `2026-09-04T13:50:38+00:00` / `2026-09-04 21:50:38 CST`.
- Runtime: about `1.90 h` wall clock.
- Final status in `run_manifest.json`: `COMPLETED`.
- Best Ultralytics epoch: `55`, with P/R/mAP50/mAP50-95
  `0.93424 / 0.96302 / 0.96667 / 0.73339`.
- Epoch 60 values: P/R/mAP50/mAP50-95
  `0.93984 / 0.95335 / 0.96289 / 0.72650`.
- `best.pt` SHA256:
  `111ae87e2911dcf35a68cb0b7d047e75f88ab62baa7367e690324133bbf2d67d`.
- `last.pt` SHA256:
  `af65a3b478c875fbe4fae882b01ed896de3c7ac2f3a6102de4e01a470422b1e2`.

Both `best.pt` and `last.pt` were evaluated on the unchanged 897 validation
images. The same prediction files were scored under both label protocols:
manually revised D3 and mapped original D0. This keeps the Data3 correction
benefit visible while guarding against hidden-test annotation-policy mismatch.

| Checkpoint | Protocol | Macro Recall | Macro FDR | Ship R/FDR | Aircraft R/FDR | Vehicle R/FDR |
| --- | --- | ---: | ---: | --- | --- | --- |
| `best.pt` | D0 original mapped | 0.885259 | 0.185691 | 0.826996 / 0.217626 | 0.989274 / 0.033324 | 0.839506 / 0.306122 |
| `best.pt` | D3 manual revision | 0.897675 | 0.176099 | 0.842991 / 0.188849 | 0.989274 / 0.033324 | 0.860759 / 0.306122 |
| `last.pt` | D0 original mapped | 0.886076 | 0.197365 | 0.828897 / 0.228319 | 0.989824 / 0.030442 | 0.839506 / 0.333333 |
| `last.pt` | D3 manual revision | 0.894885 | 0.190603 | 0.846729 / 0.198230 | 0.989824 / 0.030442 | 0.848101 / 0.343137 |

Inference configuration for both checkpoints:

- mode: tiled;
- raw confidence: `0.10`;
- per-class threshold: `0.30` for classes `0-23`, `0.35` for class `24/FSC`;
- tile/image size: `1024`;
- tile overlap: `0.20`;
- merge IoU: `0.50`;
- YOLO IoU: `0.60`;
- tile batch: `1`;
- half precision: enabled;
- config SHA256:
  `8158784300018b9923bd1cf456c6f2af2776f1fa05306fc68a85a0fd22ac87bc`.

Timing on the RTX 5090 validation run:

| Checkpoint | Total seconds | Max image seconds | Result SHA256 | Timing SHA256 |
| --- | ---: | ---: | --- | --- |
| `best.pt` | 46.274 | 4.284 | `11e7e0202bbc18bcd114a192cfa8993989adc7ff254f1cdd5d5e6edca6a5a675` | `7b8107528e861f131ccb59981bafeadb08b4e8e73a766d8c3c1dabb9078f98ec` |
| `last.pt` | 46.980 | 4.344 | `edaf846519eceeab771b2be2e52b0e83e767b1a5658139085a7fd6e5edb4225c` | `e96e99032177aa2163c42aa3052a9b4437e7cde0481d3d4a343c928586438151` |

Decision: freeze `best.pt` for the next packaging and RTX 3090 validation step.
It has the stronger worst-protocol FDR margin. `last.pt` gains only `+0.000817`
worst-protocol Recall versus `best.pt`, while its worst-protocol FDR worsens from
`0.185691` to `0.197365`, almost touching the hard `0.20` gate. The retained
local safety warning is important: `best.pt` passes both internal protocol gates,
but it does not meet the stricter planning target of Recall `>= 0.90` and FDR
`<= 0.12`.

Relative to the strongest formal hidden-test baseline `v1.0`, EXP006 best is a
promising local candidate but not yet a guaranteed submission:

- official `v1.0` hidden macro Recall/FDR: `0.844112 / 0.227895`;
- EXP006 best worst local Recall/FDR: `0.885259 / 0.185691`;
- apparent local margin: `+0.041147` Recall and `-0.042204` FDR.

Because previous candidates shifted strongly between local validation and the
formal platform, this comparison must be treated as evidence for RTX 3090
packaging, not as an official-score prediction.
