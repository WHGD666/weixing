# EXP006_X0_best Submission Candidate

Status: **FORMAL v4.0 submitted and rejected by hidden rigid gates; retained as
negative platform-shift evidence**

## Identity

- Candidate name: `EXP006_X0_best`
- Source experiment: `EXP006_X0_data3_yolo11x_1024`
- Source model: Data3 + official YOLO11x, 1024 px, 60 epochs, RTX 5090
- Selected checkpoint: `best.pt` from epoch 55
- Rejected checkpoint: `last.pt`, because its FDR margin is too thin
- Model SHA256:
  `111ae87e2911dcf35a68cb0b7d047e75f88ab62baa7367e690324133bbf2d67d`

## Internal Evidence

The same 897-image prediction was scored under both D0 original-mapped labels
and D3 manual-revision labels. These are internal validation results, not
official hidden-test scores.

| Protocol | Macro Recall | Macro FDR | Ship R/FDR | Aircraft R/FDR | Vehicle R/FDR |
| --- | ---: | ---: | --- | --- | --- |
| D0 original mapped | 0.885259 | 0.185691 | 0.826996 / 0.217626 | 0.989274 / 0.033324 | 0.839506 / 0.306122 |
| D3 manual revision | 0.897675 | 0.176099 | 0.842991 / 0.188849 | 0.989274 / 0.033324 | 0.860759 / 0.306122 |

Inference config: tiled, `imgsz=1024`, `conf=0.10`, class thresholds
`0.30` for classes `0-23` and `0.35` for `FSC`, `tile_overlap=0.20`,
`merge_iou=0.50`, `iou=0.60`, `tile_batch=1`, FP16.

RTX 5090 validation timing: `46.274 s` total for 897 images, max image
`4.284 s`. RTX 3090 timing is still required before formal submission.

## Local Validation Status

- Static Python/test suite: passed, 9 tests.
- `submit/app/main.py` Python smoke: passed on 12 images with local RTX 4060.
- Smoke timing: `8.495 s` total, max image `7.641 s`.
- Docker build: passed, image `weixing-submission:x6`.
- Docker image ID:
  `sha256:31d57f5020ee892b726b89498e5b687164c784e7c26dd6036451371d4cae29a1`.
- Docker no-network smoke: passed on 12 images, `2.865 s` total, max image
  `1.872 s`.
- Docker no-network full validation: passed on 897 images, `92.741 s` total,
  max image `1.929 s`.
- Full Docker output SHA256:
  `0b1c1851acb838dddf07b69fb1c9f2a182672ab31b1abf456dd9e7a28e83e809`.
- Full Docker timing SHA256:
  `7d2b1592d86a7bcdb834978ecd24b053066149d305436b304c79f20ae43b4b40`.
- Docker full D0/D3 scoring passed when evaluated by actual output coverage
  order. The strict frozen-order command failed because the container reads an
  input directory in sorted filename order; this is an evaluation harness order
  assumption, not a result-format or container failure.

Docker full metrics:

| Protocol | Macro Recall | Macro FDR | Ship R/FDR | Aircraft R/FDR | Vehicle R/FDR |
| --- | ---: | ---: | --- | --- | --- |
| D0 original mapped | 0.885167 | 0.185694 | 0.826996 / 0.217626 | 0.988999 / 0.033333 | 0.839506 / 0.306122 |
| D3 manual revision | 0.897583 | 0.176102 | 0.842991 / 0.188849 | 0.988999 / 0.033333 | 0.860759 / 0.306122 |

## Operator Commands

Run from `D:\daima\weixing\overshiyan` after Docker Desktop is running.

```powershell
docker build -f submit/Dockerfile -t weixing-submission:x6 .
```

```powershell
docker run --rm --gpus all --network none `
  -v "${PWD}\submit\test-input:/input:ro" `
  -v "${PWD}\submit\test-output\docker_smoke_20260905:/output" `
  weixing-submission:x6 --input /input --output /output
```

Full 897-image validation can mount the prepared validation view directly:

```powershell
docker run --rm --gpus all --network none `
  -v "${PWD}\workspace\data3_exp006\images\val:/input:ro" `
  -v "${PWD}\submit\test-output\docker_full_20260905:/output" `
  weixing-submission:x6 --input /input --output /output
```

Then score the full Docker output with the same D0/D3 protocols:

```powershell
python scripts/05_evaluate_protocols.py `
  --predictions submit/test-output/docker_full_20260905/result.json `
  --timings submit/test-output/docker_full_20260905/timings.json `
  --output-dir submit/test-output/docker_full_20260905/metrics_allow_order `
  --allow-sample
```

Use the next formal tag displayed by the platform page. After `v1.0-v3.0`, the
expected next tag is `v4.0`.

```powershell
docker tag 31d57f5020ee competition-registry.cn-beijing.cr.aliyuncs.com/competition/team614651:v4.0
docker push competition-registry.cn-beijing.cr.aliyuncs.com/competition/team614651:v4.0
```

Do not commit registry passwords, temporary login tokens, Docker layers, or model
weights. Record the pushed image digest and platform result separately.

## Official Formal Result

Platform formal `v4.0` was submitted and completed as submission `4312`.

| Group | Recall | FDR |
| --- | ---: | ---: |
| ship | 0.570801 | 0.387526 |
| aircraft | 0.959279 | 0.052816 |
| vehicle | 0.852632 | 0.372093 |

Derived from the visible platform fields:

- Score: `67.0379`
- Score time: `2026-09-05 09:24:38`
- Macro Recall: `0.794237`
- Macro FDR: `0.270812`
- Average inference time: `3.3346 s`
- Rigid gates: Recall failed, FDR failed, time passed

Decision: reject this candidate as a direct replacement for formal `v1.0`.
Compared with the local Docker D0/D3 worst result, the official hidden result
lost about `0.090930` macro Recall and gained about `0.085118` macro FDR. The
shift was concentrated in the ship group: ship Recall fell to `0.570801` and
ship FDR rose to `0.387526`. Aircraft stayed stable, while vehicle FDR remained
above the rigid gate.

The final remaining formal opportunity should not reuse this direct
Data3 + YOLO11x package without a major corrective reason. Full official
feedback is archived at
`shiyan/submissions/official_feedback/FORMAL004_v4_4312.md`.
