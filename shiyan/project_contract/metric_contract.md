# Metric Contract v0

状态：已实现；线上评分公式见 `score_contract.md`。

实现入口：`shiyan/scripts/evaluate_official.py`

## 主要指标

```text
Recall = TP / (TP + FN)
FDR = FP / (FP + TP)
```

## 匹配规则

- 预测框按 score 从高到低匹配。
- 每个预测框最多匹配一个真实框。
- 每个真实框最多被一个预测框匹配。
- 重复框计为 FP。
- 子类 ID 必须正确。

## IoU 阈值

| 类型 | IoU |
| --- | ---: |
| vehicle / FSC | 0.35 |
| ship / aircraft | 0.50 |

## 必须输出的本地指标

- 25 类 TP、FP、FN、Recall、FDR。
- ship、aircraft、vehicle 三大类 Recall、FDR。
- `group_mean`：先按三大类汇总 TP、FP、FN，再对三大类 Recall/FDR 做算术平均；这是历史报告保持兼容的三大类门槛候选。
- `group_class_macro` / `three_group_macro`：先计算各细分类别 Recall/FDR，再在各大类内和三大类之间做宏平均；用于对应评分文件中“子类先平均”的解释。
- 最大尺寸大图推理时间；本地 JSON 中的逐图耗时只作内部代理，不能冒充平台隐藏大图的 `average_inference_time_sec`。

## 刚性门槛口径

- `overall` 是所有类别 TP、FP、FN 汇总后的 pooled 诊断指标，用于观察整体表现，不作为刚性门槛的唯一依据。
- `evaluate_official.py` 同时输出 `gate_candidates`：`group_pooled_mean` 对应历史兼容口径，`group_class_macro_mean` 对应子类宏平均口径，`pooled_overall` 仅作诊断。
- 历史 `gates` 字段和 `group_mean` 字段保持不变，便于复核旧实验；在比赛方进一步明确聚合层级前，不删除任何候选口径。
- 看到本地 gate 为 `true` 只能说明公开数据上的内部复现通过，不能保证隐藏测试集或晋级排名。
- 该脚本是公开规则基础上的内部复现，不等同于比赛方封闭测试评测器；隐藏测试集上的最终分数仍以平台结果为准。

## 线上综合分

平台更新公示的 Recall/FDR/Inference Time 分段得分和历史官方分数校验，见 [`score_contract.md`](score_contract.md)。本地评估使用三大类平均 Recall/FDR 生成同口径的内部公式复现，不把它写成官方隐藏测试成绩。
