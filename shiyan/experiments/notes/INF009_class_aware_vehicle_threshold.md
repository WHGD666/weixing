# INF009：车辆类别感知置信度阈值对照

状态：**in_progress，尚未产生用户执行结果**

这是第二轮提交实验的第一组实验。目标是在不重新训练、不修改原始标签、不改变验证集和不改变模型输出框的前提下，只对车辆大类中的 `FSC`（类别 24）做离线置信度过滤，验证能否降低车辆误检率。

## 1. 实验假设

官方试运行中车辆 Recall 为 `0.939597`，但车辆虚警率为 `0.274611`，明显是当前最值得优先处理的风险点。若提高 `FSC` 的类别专属阈值，可以减少低置信度车辆误检，同时保持大部分车辆 TP 和其他类别结果不变。

## 2. 固定条件

| 项目 | 固定值 |
| --- | --- |
| 控制版本 | `SUB001` / `submit/v1` / `trial-v1.0` |
| 模型 | `EXP001` 的 `best.pt`，不变 |
| 权重 SHA-256 | `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |
| 标签 | `v0_original`，不变 |
| 验证划分 | `v1_scene_80_20`，897 张，不变 |
| 源预测 | `runs/test/SUB001_tiled_val_conf030_nms060_20260829/result.json` |
| 源参数 | tiled，1024 / 0.20 / 0.50，global conf 0.30，NMS IoU 0.60 |
| 改变因素 | 仅类别 24 `FSC` 的 score threshold |
| 其他 24 类 | 保持源预测原样 |
| 重训 | 不进行 |
| 原始标签修改 | 不进行 |

源预测已经使用全局 `conf=0.30`，因此低于 0.30 的类别阈值不会产生新结果。本轮只测试 `0.30、0.35、0.40、0.45` 四个候选，最多生成 3 个非控制变体。

## 3. 本地控制结果

以下为源预测对应的第一轮本地固定验证结果，性质是内部验证，不是官方封闭测试成绩：

| 指标 | 控制值 |
| --- | ---: |
| Overall Recall | 0.965590 |
| Overall FDR | 0.070975 |
| ship Recall / FDR | 0.832700 / 0.200730 |
| aircraft Recall / FDR | 0.987899 / 0.044681 |
| vehicle Recall / FDR | 0.827160 / 0.343137 |

## 4. 候选和选择规则

| 候选 | FSC threshold | 说明 |
| --- | ---: | --- |
| control | 0.30 | 源预测原样 |
| INF009-A | 0.35 | 过滤 FSC 低分框 |
| INF009-B | 0.40 | 过滤 FSC 低分框 |
| INF009-C | 0.45 | 过滤 FSC 低分框 |

候选必须先通过结果 schema 和图片覆盖率检查，然后使用相同的 `evaluate_official.py` 评估。采用以下选择门槛：

- Overall Recall 相对控制值下降不超过 `0.005`；
- Overall FDR 不高于控制值 `0.070975`；
- 三大类 Recall 相对控制值下降不超过 `0.010`；
- vehicle FDR 必须相对控制值 `0.343137` 有明确下降；
- Recall、FDR、Latency 三个内部 gate 全部为 true。

如果没有候选同时满足这些条件，INF009 判为未采用，保留 `v1`；不为了降低车辆 FDR 而牺牲整体 Recall。

## 5. 用户执行命令

从仓库根目录、已激活的 `(weixing)` 环境运行。以下命令只读取已有 `result.json`，不会调用模型推理：

```powershell
$source = "runs/test/SUB001_tiled_val_conf030_nms060_20260829"
$thresholds = @(0.30, 0.35, 0.40, 0.45)

foreach ($threshold in $thresholds) {
  $label = "fsc" + ("{0:00}" -f [int]($threshold * 100))
  $output = "runs/test/INF009_${label}_tiled_20260829"
  python .\shiyan\scripts\filter_class_thresholds.py `
    --input "$source/result.json" `
    --image-list "$source/image_list.txt" `
    --timings "$source/timings.json" `
    --output-dir $output `
    --class-threshold "24=$threshold"
}
```

然后评估四个候选：

```powershell
foreach ($threshold in $thresholds) {
  $label = "fsc" + ("{0:00}" -f [int]($threshold * 100))
  $output = "runs/test/INF009_${label}_tiled_20260829"
  python .\shiyan\scripts\evaluate_official.py `
    --predictions "$output/result.json" `
    --image-list "$output/image_list.txt" `
    --timings "$output/timings.json" `
    --output-dir "$output/metrics"
}
```

## 6. 结果记录要求

用户执行完成后，把四个候选的以下内容发回：预测框总数、Overall TP / FP / FN、Overall Recall、Overall FDR、三大类 Recall/FDR、三项 gate，以及每个候选的 `metrics/official_metrics.md` 路径。收到结果后再补充本记录、run registry 和采用/淘汰决策。

本轮不修改 `submit/v1`，不创建 `submit/v2`，不进行 Docker 构建，也不进行官方提交。只有候选在固定验证集上通过选择规则后，才进入 Docker 全量复核。
