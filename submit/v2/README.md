# submit/v2：FSC 类别阈值候选交付包

状态：**Windows 全量验证通过，等待 Docker 验证，尚未官方提交**

这是基于 `submit/v1` 的第二轮交付候选。唯一行为变化是：在全部 tiled 检测和跨窗口 NMS 完成后，对类别 24 `FSC` 使用 score threshold `0.35`；其他类别、模型权重、输入输出接口和推理参数保持不变。

## 固定配置

| 项目 | 值 |
| --- | --- |
| 基线版本 | `submit/v1` / `trial-v1.0` |
| 选择实验 | `INF009-fsc35` |
| 模型 | `models/best.pt` |
| 模型 SHA-256 | `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |
| 推理模式 | `tiled` |
| tile-size / overlap | `1024 / 0.20` |
| merge-iou | `0.50` |
| 全局 conf / NMS IoU | `0.30 / 0.60` |
| FSC category ID / threshold | `24 / 0.35` |
| max-det / tile-batch | `300 / 4` |

## 改动边界

- 不重新训练模型；
- 不修改原始标签；
- 不修改 `submit/v1`；
- 不改变 25 类类别顺序；
- 只过滤类别 24 且 score 小于 `0.35` 的预测框；
- 默认接口仍为 `python /app/main.py --input /input --output /output`。

## 验证顺序

1. 运行 `python -m py_compile` 和 `python submit/v2/app/main.py --help`；
2. 使用 v1 的 12 张 smoke 输入运行 v2，确认输出 schema 和 FSC 过滤行为；
3. 使用固定 897 张验证图片运行 v2，调用本地 `evaluate_official.py`；
4. 构建 `linux/amd64` Docker 镜像；
5. 在 GPU、`--network none` 下完成 Docker smoke 和全量复核；
6. 只有全部复核通过后，才考虑推送 `trial-v2.0`。

## 本地候选依据

INF009 固定验证集结果中，`fsc35` 的 Overall Recall 为 `0.965590`，Overall FDR 为 `0.070554`，Vehicle Recall 为 `0.827160`，Vehicle FDR 为 `0.330000`。这些是公开固定验证集的内部结果，不是平台隐藏测试成绩。

## Windows 入口全量验证

使用与 v1 相同的 897 张固定验证图片和 tiled 参数，v2 入口输出与 INF009-fsc35 离线过滤结果完全一致：

| 指标 | 结果 |
| --- | ---: |
| Images | 897 |
| TP / FP / FN | 4097 / 311 / 146 |
| Overall Recall | 0.965590 |
| Overall FDR | 0.070554 |
| Group-mean Recall | 0.882586 |
| Group-mean FDR | 0.191804 |
| Total seconds | 34.364042 |
| Max image seconds | 4.010357 |
| 三项 gate | 全部通过 |

证据目录：`submit/v2/test-output/app_full_20260830/`。下一步只做 Docker smoke 和 Docker 全量复核。
