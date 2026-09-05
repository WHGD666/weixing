# EXP006_X0_best Submission Candidate

Status: **frozen for validation; not yet pushed as a formal platform image**

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
- Docker build: not run because Docker Desktop was not running locally.
- Docker no-network smoke/full validation: pending.

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
  --output-dir submit/test-output/docker_full_20260905/metrics
```

Use the next formal tag displayed by the platform page. After `v1.0-v3.0`, the
expected next tag is `v4.0`.

```powershell
docker tag [IMAGE_ID_FOR_weixing-submission:x6] competition-registry.cn-beijing.cr.aliyuncs.com/competition/team614651:v4.0
docker push competition-registry.cn-beijing.cr.aliyuncs.com/competition/team614651:v4.0
```

Do not commit registry passwords, temporary login tokens, Docker layers, or model
weights. Record the pushed image digest and platform result separately.
