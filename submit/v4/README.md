# submit/v4：EXP004 原始标签继续训练 + control 后处理

这是正式平台 `v3.0`（提交 ID `3872`）的冻结 Docker 构建来源，不覆盖 `submit/v2` 或 `submit/v3`。本目录中的 `v4` 是项目内部目录编号，不等于平台自动生成的 tag。

## 固定候选配置

| 项目 | 值 |
| --- | --- |
| 模型来源 | `EXP004_original_continue40/weights/best.pt` |
| 标签版本 | `v0_original` |
| 推理模式 | `tiled` |
| 全局 conf / NMS IoU | `0.10 / 0.60` |
| tile-size / overlap | `1024 / 0.20` |
| merge-iou / tile-batch | `0.50 / 4` |
| 船只类别 0-3 阈值 | `0.30` |
| 飞机类别 4-23 阈值 | `0.30` |
| FSC 类别 24 阈值 | `0.35` |

control 配置不依赖图像颜色分类；船只无论灰度图还是彩色图都使用 `0.30`，因此入口行为是确定的。

## 容器接口

平台只需注入：

```text
--input /input --output /output
```

其余参数均有上述固定默认值，也可以在本地测试时显式传入。

## 本地测试顺序

```powershell
docker build --platform linux/amd64 -t weixing-submission:v4 submit/v4
```

使用 `--network none` 做 12 张 smoke：

```powershell
New-Item -ItemType Directory -Force "submit/v4/test-output/docker_smoke" | Out-Null
docker run --rm --gpus all --network none `
  -v "${PWD}\submit\v4\test-input-smoke:/input:ro" `
  -v "${PWD}\submit\v4\test-output\docker_smoke:/output" `
  weixing-submission:v4 `
  --input /input --output /output
```

该流程已完成：smoke、固定 897 张 Docker 全量运行、schema 与内部指标检查均通过，随后镜像以平台 tag `v3.0` 推送并完成正式评测。

Docker 入口会按文件名排序输入图片。全量评估时，必须使用与该排序一致的验证清单，不能直接使用原始 `val.txt`：

```powershell
$stage = Join-Path $PWD "tmp\EXP004_control_val_staging_20260902"
$orderedList = Join-Path $PWD "tmp\EXP004_control_val_ordered_20260902.txt"
$labelRoot = Join-Path $PWD "shiyan\data\images\train"
Get-ChildItem $stage -File |
  Sort-Object Name |
  ForEach-Object { Join-Path $labelRoot $_.Name } |
  Set-Content $orderedList -Encoding ascii
```

然后将评估器的 `--image-list` 指向 `$orderedList`。该清单只改变评估时的顺序，不改变 897 张验证图片、标签或模型。

## 证据边界

本目录的本地结果只能作为内部验证，不能写成官方隐藏测试成绩。最终是否通过刚性指标，以比赛后台正式提交结果为准。

正式结果：ship R/FDR `0.697666/0.364222`，aircraft `0.957740/0.058617`，vehicle `0.884211/0.363636`，三大类宏平均 `0.846539/0.262158333`，时间 `1.742 s`，综合分 `69.8637`。完整复盘见 [`FORMAL003_v3_3872.md`](../../shiyan/submissions/official_feedback/FORMAL003_v3_3872.md)。
