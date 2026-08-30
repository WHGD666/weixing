# INF010：v3 最终后处理强化实验

状态：**completed，未产生可交付 v3；下一步转入数据集审计**

这是数据集修正前的最后一轮后处理探索。目标是在不重新训练、不修改原始标签、不改变验证划分的前提下，围绕 v2 已验证的车辆误检改善方向做一次有边界的强化实验。若没有清晰且可复现的增益，本实验结束后转入数据集审计与标签修正，不再继续无依据地扫参数。

## 1. 冻结对照

| 项目 | 固定值 |
| --- | --- |
| 官方对照 | `submit/v2` / `trial-v2.0` / 官方提交 `2750` |
| 当前官方分数 | `84.8310`，排名第 42 |
| 模型 | EXP001 `best.pt`，SHA-256 `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |
| 验证划分 | `v1_scene_80_20`，897 张 |
| 本地对照预测 | `submit/v2/test-output/docker_full_20260830/result.json` |
| 对照参数 | tiled，1024 / 0.20 / merge IoU 0.50，global conf 0.30，NMS IoU 0.60，FSC conf 0.35 |
| 标签 | `v0_original`，只读 |
| 重训 | 不进行 |
| holdout | 未评估，不能声称泛化到隐藏测试集 |

官方 v2 相对 v1 的变化已记录在 `SUB002_trial_2750.md`：vehicle FDR 从 `0.274611` 降到 `0.244565`，vehicle Recall 从 `0.939597` 降到 `0.932886`，总分提升 `0.4997`。因此 v3 只围绕车辆误检和候选框重复风险分析，飞机参数先固定。

## 2. 本轮边界

1. 第一阶段只运行只读诊断，统计类别和分数区间的 TP/FP，以及同类高重叠框的组合。
2. 诊断后最多生成 3 个离线候选；不在没有证据时同时改多个互相耦合的参数。
3. 每个候选必须使用同一份 897 张验证集、同一 `evaluate_official.py` 和同一门槛评估。
4. 在候选选定前不修改 `submit/v2`，不构建新 Docker，不推送仓库，不提交官方平台。

## 3. 预设候选方向

候选方向按诊断结果决定，优先级如下：

- FSC/vehicle 的分数阈值或分数区间过滤，延续 INF009 已验证方向；
- 若近重复框证据充分，只调整同类合并/NMS 的一个参数；
- 只有在数据明确显示 ship 低分误检可安全处理时，才考虑 ship 的单类阈值；aircraft 保持不变。

## 4. 选择与停止规则

相对 v2 本地对照，候选必须同时满足：

- 结果 schema 和图片覆盖率正确；
- 三大类 macro Recall/FDR/Latency gate 全部通过；
- group-mean Recall 下降不超过 `0.005`；
- 任一大类 Recall 下降不超过 `0.010`；
- 声明的目标指标至少有 `0.01` 的绝对改善，且没有明显的非目标组回退；
- 候选必须优于 v2 对照，否则不进入 Docker 和官方验证。

如果最多 3 个候选都没有清晰改善，停止后处理探索，保留 v2 作为可回滚版本，转入 `AUDIT002` 数据集审计。所有判断都以本地公开验证集为依据，不能把本地 gate 当作官方隐藏测试通过证明。

## 5. 当前待执行命令

从仓库根目录、已激活的 `(weixing)` 环境运行：

```powershell
$v2 = "submit/v2/test-output/docker_full_20260830"
$diag = "runs/test/INF010_v2_diagnostic_20260830"
New-Item -ItemType Directory -Force $diag | Out-Null

python .\shiyan\scripts\analyze_confidence_bins.py `
  --predictions "$v2/result.json" `
  --image-list "submit/v1/test-output/app_full_20260829/image_list_original.txt" `
  --output-dir $diag `
  --duplicate-iou 0.30
```

诊断输出：

- `confidence_bins.csv`：每个类别、每个分数区间的预测数、TP、FP、FDR；
- `near_duplicate_summary.csv`：同类 IoU 不低于 0.30 的近重复框及 TP/FP 组合；
- `diagnostic_summary.json`：输入、阈值和输出清单。

## 6. 诊断结果

诊断使用 v2 Docker 全量结果，覆盖固定验证集 897 张图、4408 个预测框：

- FSC 类别共有 100 个预测框；低于 `0.60` 的框同时包含真阳性和误检，不能安全地整体删除；
- 同类 IoU≥`0.30` 的近重复对共 35 对，其中 FSC 只有 6 对：3 对为 `FP-FP`，3 对为 `TP-TP`，没有明显的 `TP-FP` 重复清理机会；
- 误检主要集中在 MS 和 FSC 的中低分段，但 FSC 的高阈值处理会牺牲真实车辆召回。

证据目录：`runs/test/INF010_v2_diagnostic_20260830/`。其中 `confidence_bins.csv` 和 `near_duplicate_summary.csv` 为只读诊断输出。

## 7. 强化候选结果

在 v2 固定预测上只测试类别 24 `FSC` 的更高阈值 `0.50、0.55、0.60`，没有重新推理、重训或修改标签：

| 候选 | Overall Recall | Overall FDR | Group-mean Recall | Group-mean FDR | Vehicle Recall | Vehicle FDR | 决策 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v2 control / FSC 0.35 | 0.965590 | 0.070554 | 0.882586 | 0.191804 | 0.827160 | 0.330000 | 对照 |
| INF010-fsc50 | 0.964648 | 0.068078 | 0.866125 | 0.165137 | 0.777778 | 0.250000 | 淘汰：车辆 Recall 下降 |
| INF010-fsc55 | 0.964648 | 0.067654 | 0.866125 | 0.159039 | 0.777778 | 0.231707 | 淘汰：车辆 Recall 下降 |
| INF010-fsc60 | 0.964176 | 0.067259 | 0.857895 | 0.154453 | 0.753086 | 0.217949 | 淘汰：车辆 Recall 下降 |

三者虽然降低了 FDR，但均违反本实验预设的 group-mean Recall 和单组 Recall 保护规则；没有候选达到可交付标准。因此不构建 v3 Docker、不推送新 tag、不提交官方平台，继续保留官方已验证的 `submit/v2` / `trial-v2.0`。

## 8. 决策与下一步

INF010 的结论是：后处理方向确实能在小范围内降低车辆误检，但单纯继续抬高 FSC 阈值已经进入明显的 Recall-FDR 交换区间，近重复框也不足以支持一次有把握的 NMS 改动。后续工作转为 `AUDIT002` 数据集审计，优先检查 FSC/MS 误检集中图像、漏标和类别边界问题；审计期间原始标签只读，任何修改都要单独形成版本和复核证据。
