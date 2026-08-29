# INF005：原始数据基线全量切片候选验证

状态：completed

## 实验目的

在第一轮不进行人工审计和标签干预的前提下，验证提交候选参数在完整固定验证集上的切片推理表现。本实验只验证工程路径和候选配置，不引入新的模型训练变量。

## 配置

| 项目 | 值 |
| --- | --- |
| 模型 | `EXP001` 的 `best.pt` |
| 标签 | 比赛方原始标签，`v0_original` |
| 验证集 | `v1_scene_80_20`，897 张 |
| 推理模式 | tiled |
| tile-size | 1024 |
| tile-overlap | 0.20 |
| merge-iou | 0.50 |
| tile-batch | 4 |
| conf | 0.30 |
| NMS IoU | 0.60 |

## 用户执行结果

- validated_images：897
- validated_objects：4410
- total_seconds：24.922 秒
- max_image_seconds：3.538 秒
- result：`runs/test/SUB001_tiled_val_conf030_nms060_20260829/result.json`
- image_list：`runs/test/SUB001_tiled_val_conf030_nms060_20260829/image_list.txt`
- 输出覆盖率：通过

## 内部指标核验

- Overall TP / FP / FN：4097 / 313 / 146
- Overall Recall：0.965590
- Overall FDR：0.070975
- ship Recall/FDR：0.832700 / 0.200730
- aircraft Recall/FDR：0.987899 / 0.044681
- vehicle Recall/FDR：0.827160 / 0.343137
- latency gate：通过

以上结果是本地固定验证集内部评估，不是比赛封闭测试成绩。整体门槛通过，但舰船大类指标较弱，作为后续模型改进和报告中的风险记录保留。

与 direct 全量候选测试相比，切片模式总耗时和最大单图耗时均更低；最终是否采用 tiled 仍以同口径 Recall/FDR 和 Docker 验证为准。

## 下一步

评估已完成。当前候选可以进入 Docker 交付目录整理，不再进行人工审计或全局阈值扫描。
