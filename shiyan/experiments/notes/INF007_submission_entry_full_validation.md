# INF007：submit/v1 交付入口全量集成验证

状态：completed

## 目的

让 `submit/v1/app/main.py` 直接处理完整的 `v1_scene_80_20` 验证图片，确认交付入口与已验证的实验推理流程在全量输入下能够正常运行。

## 配置

- 入口：`submit/v1/app/main.py`
- 模型：`runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/best.pt`
- 输入：`submit/v1/test-input/`，由固定验证清单复制得到
- 图片数：897
- 推理模式：tiled
- tile-size：1024
- tile-overlap：0.20
- merge-iou：0.50
- tile-batch：4
- conf：0.30
- NMS IoU：0.60

## 用户执行结果

- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- images：897
- total_seconds：40.793 秒
- max_image_seconds：5.049 秒
- result：`submit/v1/test-output/app_full_20260829/result.json`
- 入口内部 result schema 校验：通过

此前实验脚本的全量切片结果为总耗时 `24.922s`、最大单图 `3.538s`。两次输入组织和运行状态不同，速度差异暂不作结论；本次交付入口结果已经完成同口径指标核验。

## 内部指标核验

- `recall_ge_0_85`：true
- `fdr_le_0_20`：true
- `latency_le_20s`：true
- Overall TP / FP / FN：4097 / 313 / 146
- Overall Recall：0.965590
- Overall FDR：0.070975
- ship Recall/FDR：0.832700 / 0.200730
- aircraft Recall/FDR：0.987899 / 0.044681
- vehicle Recall/FDR：0.827160 / 0.343137
- max image time：5.049 秒
- 详细 Recall、FDR、TP、FP、FN：以 `submit/v1/test-output/app_full_20260829/metrics/official_metrics.md` 为准。

三项标志全部通过，说明交付入口与固定验证协议已经接通。该结果仍然是本地内部验证，不是官方封闭测试成绩。

## 下一步

评估已完成。下一阶段进入 Dockerfile 放入和环境交付准备。
