# EXP005：data3 人工修订与类别映射修正版

## 状态

训练与 D0/D3 双口径验证已完成；直接作为提交模型被拒绝，尚未进行官方隐藏测试。

## 实验目的

重新验证人工修订标签的价值，并修复上一轮导出过程中发现的类别字典问题。为便于归因，本实验继续以第一次正式提交所用的原始模型 `submit/v2/models/best.pt` 为初始化权重，不使用 EXP002 人工修订模型或 EXP004 续训模型。

## 数据审计

- 数据目录：`shiyan/data3`
- 图片：4481
- 标签：4481，其中 4 个确认无目标的空标签为 `0581/0582/0583/0910`
- 目标总数：21008
- 类别数：25
- 训练前审计快照指纹：`9e330f03b56c3728ed1c485404d92306ed94edb22c7bcd8a0f09ac0a91438fa8`（不再代表当前磁盘标签）
- 审计错误：0
- 审计警告：47，其中 4 个为空标签，43 个框轻微越过图像边缘；切片边界目标允许保留，训练前不自动裁改人工标签
- 类别顺序：`22=A19_SU-34`，`23=A20_SU-24`

完整审计见 `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/audit_report.md`。

## 固定划分

沿用原始数据的 `v1_scene_80_20` 冻结划分，并通过重命名清单映射到 data3：

- 训练集：3584 张
- 验证集：897 张
- 交集：0
- 覆盖：4481 张

## 训练设计

- 最大轮数：150
- 初始化模型：`submit/v2/models/best.pt`
- 图像尺寸：1024
- batch：4
- 固定种子：20260902
- 前 80 轮：禁止早停
- 第 81 轮起：重新建立早停基线，连续 30 轮本地 Ultralytics fitness 不提升则停止
- 每 10 轮保存恢复点
- 最终模型选择：训练目录中的 `weights/best.pt`，不是固定使用最后一轮
- 实际输出目录：`runs/detect/runs/train/exp005_data3_manual_revision_classfix-2`

早停用于限制第 80 轮之后的验证集过拟合，不代表官方三大硬指标；训练完成后仍须单独计算三大类 Recall、FDR 和推理时间。

## 启动命令

仅检查训练前条件：

```powershell
python shiyan/scripts/train_exp005_data3.py --preflight-only
```

正式训练：

```powershell
python shiyan/scripts/train_exp005_data3.py
```

中断后恢复：

```powershell
python shiyan/scripts/train_exp005_data3.py --resume runs/detect/runs/train/exp005_data3_manual_revision_classfix/weights/last.pt
```

## 结果登记

- 实际完成轮数：150
- 总训练时间：约 5.712 小时
- 最佳 epoch：137
- `best.pt`：`runs/detect/runs/train/exp005_data3_manual_revision_classfix-2/weights/best.pt`
- `best.pt` SHA256：`3D3C8CB2B256B127668EC113F50DA7C261A295C30C7AEB2A2E177D2682D5A0E5`
- best epoch Ultralytics 指标：Precision `0.89012`、Recall `0.90344`、mAP50 `0.92466`、mAP50-95 `0.72568`
- 训练扫描自动删除重复标签：训练图 `1356.jpg` 1 条、验证图 `1344.jpg` 1 条；后续审计保留该事实
- 上述指标以 D3 验证标签计算，不是三大类硬指标，也不是官方隐藏测试成绩

## 双标签验证协议

D0 与 D3 的验证清单均为 897 张，已通过逐图映射核验：图像 SHA256、尺寸和顺序全部一致。标签语义比较结果：

- 数值发生变化的标签文件：560
- 目标数量或类别直方图发生变化的文件：38
- 同类别框在 IoU `>=0.50` 下完全等价的文件：850
- 原始验证目标数：4243
- 当前 D3 验证目标数：4251
- 训练时 Ultralytics 删除验证图 `1344.jpg` 的 1 条重复标注，因此训练日志显示 4250 个验证实例
- D3 标签 manifest 有 8 个文件的目标数记录与当前磁盘内容不一致；训练前登记的数据指纹不能继续代表当前 data3 状态

因此 EXP005 必须同时评估：

1. D0 原始标签口径，用于与 Model A、Model B 和正式提交历史横向比较；
2. D3 人工修订标签口径，用于判断模型是否学到修正后的目标定义；
3. 两套结果不得混写，候选选择优先查看两套口径中的最差 Recall/FDR。

当前协议核验证据：`runs/test/INF012_protocol_mapping_final_20260904/`。现有 EXP005 权重可以继续做只读验证，但任何后续 data3 重训必须先重建 manifest、审计报告和数据指纹。

## 双口径比赛指标结果

Model C 使用统一低阈值缓存和固定 control 阈值完成 897 张评估：

| 协议 | 三大类宏 Recall | 三大类宏 FDR | 最大单图代理时间 |
| --- | ---: | ---: | ---: |
| D0 原始标签 | 0.798630 | 0.228070 | 0.086787 s |
| D3 人工修订标签 | 0.808783 | 0.217170 | 0.086787 s |

D0 分组为 ship `0.777567/0.296041`、aircraft `0.976348/0.072380`、vehicle `0.641975/0.315789`。两套标签下 Recall 和 FDR 均未同时过线，因此 Model C 不直接进入提交候选，也不接管任何三大类。它只保留为人工修订标签实验事实和后续一致性诊断来源。

本结果再次说明 Ultralytics best epoch 的 Recall/mAP 不能替代比赛三大类宏平均 Recall/FDR。完整对比见 `INF012_three_model_dual_protocol_search.md`。
