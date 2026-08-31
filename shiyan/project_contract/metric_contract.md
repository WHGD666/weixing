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
- 三大类平均 Recall、FDR。
- 最大尺寸大图推理时间。

## 刚性门槛口径

- `overall` 是所有类别 TP、FP、FN 汇总后的 pooled 诊断指标，用于观察整体表现，不作为刚性门槛的唯一依据。
- 刚性 Recall/FDR gate 采用 ship、aircraft、vehicle 三大类指标的算术平均，即 `group_mean`。
- `evaluate_official.py` 同时保留 pooled gate 诊断值，但输出的 `gates` 使用三大类平均值，并明确写入 `gate_basis`。
- 该脚本是公开规则基础上的内部复现，不等同于比赛方封闭测试评测器；隐藏测试集上的最终分数仍以平台结果为准。

## 线上综合分

平台更新公示的 Recall/FDR/Inference Time 分段得分和历史官方分数校验，见 [`score_contract.md`](score_contract.md)。本地评估使用三大类平均 Recall/FDR 生成同口径的内部公式复现，不把它写成官方隐藏测试成绩。
