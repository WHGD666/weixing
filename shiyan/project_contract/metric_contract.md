# Metric Contract v0

状态：draft，等待评估脚本实现和样例校验后冻结。

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
