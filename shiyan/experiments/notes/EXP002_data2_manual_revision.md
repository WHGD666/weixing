# EXP002：data2 第一轮人工修订训练与正式提交冻结记录

状态：**frozen，正式结果已回收；不作为当前最佳模型**

本实验验证 2026-09-01 完成的第一轮人工标签微调是否能改善隐藏测试表现。用户修改了 `data2` 前 1360 张图，重点是舰船和 FSC/导弹车；本轮没有大规模重标，也没有重新训练数据集。实验已经完成训练、固定验证集测试、Docker smoke、镜像推送和平台提交，正式结果为 submission `3720`、tag `v2.0`。

## 1. 实验身份与可追溯信息

| 项目 | 值 |
| --- | --- |
| Experiment ID | `EXP002_data2_manual_revision` |
| 实验角色 | 人工修订标签候选；正式提交验证 |
| 数据目录 | `shiyan/data2`，本地工作副本，未直接提交到 GitHub |
| 数据说明 | 4481 张图片，25 类，文件名 `0001` 至 `4481` |
| 场景划分 | `v1_scene_80_20_data2`，train 3584，val 897 |
| 标签清单 | `shiyan/data2/manifests/rename_0001_4481_manifest.csv` |
| 清单 SHA256 | `59293EBD88D89138742CD12400F219E0CA918DC66DB06022B51685EA546C605B` |
| val 清单 SHA256 | `F8EEE36341820EEA59C66497737A73061C026326A14EE0CC27929807A115D1ED` |
| 训练配置 | `shiyan/configs/train/exp002_data2_manual_revision.yaml` |
| 训练配置 SHA256 | `8DC256B6F36628301E5683CD5FB4904BEF389677526B4967D1174211B2B8454C` |
| 数据集 YAML SHA256 | `77D25513233103B40F2735940EE23DE265D0008FDF97B060C62B8BC7CCFEE98D` |
| Git 基线 | `7032ed8` |

## 2. 数据修订范围

- 人工修订范围：`0001` 至 `1360`。
- 用户确认空白图：`0581`、`0582`、`0583`、`0910`。这 4 个标签文件为空文件，表示没有可用目标，不添加伪框。
- 机器验收：`ok=true images=4481 labels=4481 classes=25`。
- 与原始标签的类别行数净变化：QHS `-3`、MS `-12`、FSC `-3`，其他类别合计不变，总目标行数 `-18`。
- 类别计数发生变化的文件约 92 张。文本差异不能代表全部框的坐标微调，因为标注工具可能重写了原始文本。
- 修改意图：删除大图切割中未真正切到的目标、修正明显过大或明显错误的框，并对舰船和 FSC 做少量人工微调；没有把不同大小的民船强行归一为固定尺寸框。

## 3. 训练过程与实际模型来源

训练命令：

```powershell
yolo detect train cfg=shiyan/configs/train/exp002_data2_manual_revision.yaml
```

实际训练完成 `55/55`，未发生 OOM。模型文件：

```text
runs/detect/runs/train/exp002_data2_manual_revision/weights/best.pt
SHA256 9378660F53BF06E613B580188D724D58647B3E1BF02A7752AB8A0FA77963D8CD
```

第 55 轮 `results.csv` 记录：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.91108 | 0.79155 | 0.88510 | 0.66950 |

训练日志文件 SHA256：`0B03EE088ABE6CE6E3406F2965318EA8ACEF181D099A2C4DE1A127E3CF4DFCE9`。

### 关键配置问题

配置中虽然写有 `model: submit/v2/models/best.pt`，但同时设置了 `pretrained: false`。检查当前 Ultralytics 的模型初始化逻辑后确认：加载 `.pt` 后，在 `pretrained: false` 且非 resume 时会丢弃 checkpoint 权重。因此 EXP002 实际是**从头初始化训练**，不是在 v1/v2 旧模型上继续微调。

这意味着本实验同时改变了标签、训练轮数和初始化方式，不能把正式结果的变化全部归因于人工标签质量。下一轮必须显式使用 `pretrained: true` 或正确的 checkpoint 微调流程，并把初始化方式固定下来。

## 4. 固定验证集推理与官方口径复现

本地 tiled 推理使用 897 张固定验证图：

```text
output=runs/test/EXP002_data2_best_tiled_20260901
validated_images=897
validated_objects=4505
total_seconds=26.659
max_image_seconds=5.034
```

随后对 FSC 类使用 `fsc-conf=0.35` 过滤：源目标 4505，过滤后 4491。官方口径评估文件为：

```text
runs/test/EXP002_data2_best_fsc035_tiled_20260901/metrics/official_metrics.json
```

本地评估结果：

| 分组 | Recall | FDR | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| ship | 0.674374 | 0.281314 | 350 | 137 | 169 |
| aircraft | 0.943619 | 0.126527 | 3431 | 497 | 205 |
| vehicle | 0.625000 | 0.342105 | 50 | 26 | 30 |
| group-pooled mean | 0.747664 | 0.249982 | 3831 | 660 | 404 |

整体 TP/FP/FN 为 `3831/660/404`，整体 Recall `0.904604`、整体 FDR `0.146961`。但按三大类分组宏平均，Recall 和 FDR 均未达到刚性门槛；推理时间通过。这个结果也说明整体汇总值不能代替三组宏平均。

## 5. Docker 验收与镜像

Docker 构建过程中曾遇到 micromamba 下载流截断，原来的 `curl | tar` 直接管道解压失败。`submit/v3/Dockerfile` 已在本地候选中改为带重试的文件下载后再解压，文件 SHA256 为 `1CFD25A88CD6D3A797FECA9D8737E50964CDCB8964F1589CDA545E1EC4E78D6A`。

smoke 使用 `--gpus all --network none`，先后修正了遗漏应用必需的 `--input /input --output /output` 参数。最终通过：

```text
gpu=NVIDIA GeForce RTX 4060 Laptop GPU
images=12
total_seconds=6.243
max_image_seconds=4.887
result=/output/result.json
```

正式镜像：

```text
competition-registry.cn-beijing.cr.aliyuncs.com/competition/team614651:v2.0
image ID 384f9e6ef18f
digest sha256:384f9e6ef18f94afb2f5d25c68f1943a014ec0dd474b93882a135a748c54a775
```

## 6. 正式平台结果

平台显示 submission `3720`、初赛、tag `v2.0`、`ACCEPTED`、`已完成`。这表示容器和评分任务完成，不等于通过刚性指标或获得 70 分基础分。

| 指标 | 平台值 |
| --- | ---: |
| ship Recall | 0.284132 |
| ship FDR | 0.174486 |
| aircraft Recall | 0.901458 |
| aircraft FDR | 0.182432 |
| vehicle Recall | 0.705263 |
| vehicle FDR | 0.524823 |
| average inference time | 1.7638 s |
| score | 58.0075 |
| result time | 2026-09-02 10:55:55 |

按平台展示值计算：

```text
macro Recall = (0.284132 + 0.901458 + 0.705263) / 3 = 0.630284
macro FDR    = (0.174486 + 0.182432 + 0.524823) / 3 = 0.293914
```

因此正式隐藏测试中宏平均 Recall 和宏平均 FDR 都没有过门槛，推理时间通过。正式平台结果与本地固定验证集差异很大，尤其是 ship Recall 和 vehicle FDR，说明本地验证集不能代表隐藏测试分布，后续必须依靠更可靠的标签审计和分层验证。

## 7. 与第一次正式提交对照

`FORMAL001`（submission `3496`、v1.0）平台得分 `70.5805`，本次 `FORMAL002` 得分 `58.0075`，下降 `12.5730`。

| 指标 | FORMAL001 | FORMAL002 | 变化 |
| --- | ---: | ---: | ---: |
| ship Recall | 0.699474 | 0.284132 | -0.415342 |
| ship FDR | 0.251032 | 0.174486 | -0.076546 |
| aircraft Recall | 0.959178 | 0.901458 | -0.057720 |
| aircraft FDR | 0.061442 | 0.182432 | +0.120990 |
| vehicle Recall | 0.873684 | 0.705263 | -0.168421 |
| vehicle FDR | 0.371212 | 0.524823 | +0.153611 |
| macro Recall | 0.844112 | 0.630284 | -0.213828 |
| macro FDR | 0.227895 | 0.293914 | +0.066018 |
| average inference time | 1.7846 | 1.7638 | -0.0208 s |
| score | 70.5805 | 58.0075 | -12.5730 |

## 8. 本轮遇到的问题与排查结论

1. Docker Linux 引擎最初未启动，出现 named pipe 连接错误；启动 Docker Desktop 后恢复。
2. micromamba 下载结果偶发截断，`curl -Ls | tar` 把不完整压缩包直接交给 tar，导致 exit code 2；已在 v3 候选 Dockerfile 中改为重试下载和文件校验式解压。
3. 第一次 smoke 命令遗漏 `--input` 和 `--output`，应用正确报参数缺失；补齐后通过。
4. 容器内 Ultralytics 配置目录不可写，自动切换 `/tmp/Ultralytics`；属于警告，不影响本次结果。
5. 阿里云镜像仓库登录多次超时，且 Docker 使用了失效的本地代理端口 `172.21.255.77:7897`；网络恢复后登录和 push 成功。
6. 训练配置 `pretrained: false` 与“继续使用旧模型”的原计划不一致，导致本轮从头训练；这是最重要的实验设计问题。
7. 本地固定验证集表现不能预测隐藏测试：EXP002 本地宏 Recall `0.747664`，平台宏 Recall `0.630284`；ship/vehicle 方向差异尤其明显。
8. 训练日志中本地 Recall/mAP 上升不能直接说明平台会提升；正式门槛看三大类宏平均指标，而不是所有目标合并后的整体 Recall。

## 9. 冻结结论与下一步边界

- 冻结保留：`data2` 当前版本、EXP002 训练模型、固定划分、Docker digest、平台结果和本记录。
- `v1.0`/FORMAL001 仍是当前正式最佳基线；EXP002/v2.0 记录为失败的负向实验，不替换默认基线。
- 不删除或覆盖 `EXP002` 输出，后续报告和比较必须引用本记录中的模型哈希与镜像 digest。
- 下一轮先修正训练初始化方式，确保“旧模型微调”和“从头训练”不再混淆；然后分别比较原始标签与 data2 标签，减少一次改变的变量。
- 数据集后续工作应优先审计 ship、vehicle 的漏标、过大框、类别边界和隐藏分布代表性；不能仅凭本地整体 mAP 判断是否上传。

## 报告图片素材

- [EXP002 训练曲线](../evidence/EXP002_data2_manual_revision/results.png)
- [EXP002 验证集预测示例](../evidence/EXP002_data2_manual_revision/val_batch0_pred.jpg)
