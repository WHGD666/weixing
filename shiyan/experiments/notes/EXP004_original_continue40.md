# EXP004：原始标签模型权重继续训练 40 轮

状态：**internal_validation_completed，待 Docker 验收；未提交平台**

本实验沿着 EXP003 的颜色后处理实验并行推进，单独回答一个问题：在不使用 `data2` 人工修订标签、不加入颜色策略的前提下，使用第一次原始标签训练得到的模型作为初始权重，再训练 40 轮，模型本身是否会改善。

## 1. 实验身份与变量边界

| 项目 | 值 |
| --- | --- |
| Experiment ID | `EXP004_original_continue40` |
| 实验角色 | 原始标签模型训练改进候选 |
| 标签版本 | `v0_original` |
| 数据配置 | `shiyan/configs/dataset/weixing_v1_scene_80_20.yaml` |
| 划分版本 | `v1_scene_80_20` |
| 训练集/验证集 | 3584 / 897 张 |
| 类别数 | 25 |
| 初始权重 | `runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/last.pt` |
| 初始权重 SHA256 | `17D648E4F8AD61E7D49986616B665745B982D3ABEF17F8C17A696703875820CE` |
| 训练输出 | `runs/detect/runs/train/exp004_original_continue40/` |
| Git 基线 | `7e20b18`，运行时工作区含未提交的无关本地文件 |

本实验没有使用 `shiyan/data2`，没有使用人工修订标签，也没有应用灰度/彩色图像后处理策略。这样可以把“训练因素”和 EXP003 的“颜色后处理因素”分开比较。

## 2. 实际训练方式

执行命令：

```powershell
yolo detect train `
  model=runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/last.pt `
  data=shiyan/configs/dataset/weixing_v1_scene_80_20.yaml `
  epochs=40 `
  patience=20 `
  batch=4 `
  imgsz=1024 `
  device=0 `
  workers=4 `
  project=runs/train `
  name=exp004_original_continue40 `
  exist_ok=false `
  pretrained=true `
  optimizer=auto `
  seed=20260828 `
  deterministic=true `
  single_cls=false `
  rect=false `
  cos_lr=true `
  close_mosaic=10 `
  amp=true `
  cache=false `
  plots=true `
  save=true `
  val=true `
  verbose=true
```

需要准确区分：命令设置了 `resume=false`，因此它加载 EXP001 的 `last.pt` 权重，但使用新的一轮 40 轮训练流程和新的优化器状态；不是恢复原 EXP001 的完整优化器断点，也不是把原来的 80 轮直接续到 120 轮。这样做是有意的，用于观察“从原始模型权重重新微调 40 轮”的独立效果。

训练实际完成 `40/40`，未发生 OOM。训练输出目录为：

```text
D:\daima\weixing\runs\detect\runs\train\exp004_original_continue40\
```

## 3. 内部验证结果

以下指标是 Ultralytics 在固定本地验证集上的检测指标，不是比赛隐藏测试结果，也不是官方三大类宏平均指标。

### 最佳轮次

按 `metrics/mAP50-95(B)` 选择，最佳为第 38 轮：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.96929 | 0.94476 | 0.96653 | 0.77968 |

### 第 40 轮

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.96338 | 0.94814 | 0.96639 | 0.77947 |

验证集摘要中，最佳模型约为 Precision `0.969`、Recall `0.945`、mAP50 `0.967`、mAP50-95 `0.780`。第 40 轮 Recall 略高，但 mAP50-95 略低，因此后续推理优先使用 `best.pt`，同时保留 `last.pt` 作为可复核产物。

### 与 EXP001 原始模型比较

EXP001 的最佳内部记录为 Precision `0.94982`、Recall `0.95606`、mAP50 `0.96406`、mAP50-95 `0.77669`。EXP004 的 mAP50 和 mAP50-95 有小幅提升，Precision 提升，Recall 略低。这个变化只能说明本地验证上的整体检测质量有轻微变化，不能预先推断隐藏测试一定提升。

训练摘要中需要重点留意的类别：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| LQS | 0.991 | 0.833 | 0.835 | 0.662 |
| QHS | 0.908 | 0.789 | 0.893 | 0.605 |
| MS | 0.873 | 0.772 | 0.866 | 0.564 |
| FSC | 0.893 | 0.675 | 0.782 | 0.449 |

FSC、MS、QHS 的内部召回仍明显低于强类别，说明继续训练并不能替代针对类别、边界截断和数据分布的审计。

## 4. 产物与哈希

| 产物 | SHA256 |
| --- | --- |
| `weights/best.pt` | `90B39F94D272648781152F28EF840B553D1AF64441815522319E7461C756048B` |
| `weights/last.pt` | `77F3EFE7FBB2DBE099BD9CA52911425B6309CCB8B6E52DD008A0A18938A9E7B2` |
| `results.csv` | `9E291D03125BF178C79D265FD436C9FCAE3B4F07AB136E36F41F2AD87EEE835C` |
| `args.yaml` | `2D9221A83DA5908F5F8E93DB4AC86B1239600912A6710DA018BF1962EFA791E4` |

模型权重和完整训练目录保留在本机，不上传 GitHub。GitHub 只保存本实验的配置说明、结果摘要和哈希，避免仓库膨胀。

## 5. 与颜色策略实验的比较计划

本实验对应四格比较中的第三格：

1. 原始 v2 模型；
2. 原始 v2 + `gray027` 颜色策略；
3. EXP004 原始标签继续训练模型；
4. EXP004 继续训练模型 + `gray027` 颜色策略。

前两格属于 EXP003 的固定 v2 模型实验；第三格目前只有训练产物和 Ultralytics 内部验证；第四格尚未运行。必须先对第三格完成与比赛规则一致的 897 张验证集推理，再对同一批预测应用 `gray027`，才能判断提升来自训练还是来自后处理。

## 6. 当前结论与限制

- EXP004 训练链路成功，原始标签和原始固定划分保持不变。
- 本地内部 mAP50-95 从 EXP001 的 `0.77669` 小幅提高到 `0.77968`，但内部 Recall 没有提高，变化幅度很小。
- 训练指标不是比赛三项刚性指标；目前不能宣布 Recall、FDR、推理时间已经达标。
- `best.pt` 已冻结为本轮后续推理候选；不覆盖 EXP001，也不替代正式提交 `v1.0 / submission 3496` 这一正式基线。
- `data2 / v3 / submission 3720` 仍是冻结的负向实验，不参与本轮原始标签训练结论。
- 由于此前本地验证与隐藏测试有明显分布差异，下一步必须完成官方口径的分组宏 Recall、分组宏 FDR 和推理时间测试，之后才能决定是否制作 Docker 候选。

## 7. 下一步命令边界

下一步只做两次同口径评估，不改标签、不改模型代码：

1. 用 `EXP004.../weights/best.pt` 生成 897 张固定验证图预测，记录三大类 Recall、FDR、时间；
2. 对同一个 EXP004 预测使用 EXP003 的 `gray027` 策略，再评估一次。

两次结果完成后，和 EXP003 已记录的原始 v2、原始 v2+`gray027` 对照。若四格中任一候选只是在本地门槛附近通过，不直接提交，先保留隐藏测试分布风险和类别级安全余量。

## 8. 全量本地验收结果

用户已完成 12 张 tiled 冒烟、897 张 tiled 全量推理、EXP004 control、EXP004 `gray027` 和错误分析。三项 gate 使用项目当前的历史兼容口径：三大类分组 pooled Recall/FDR 的算术平均，以及最大单图时间。

### 8.1 12 张 tiled 冒烟

```text
validated_images = 12
validated_objects = 48
total_seconds = 6.164 s
max_image_seconds = 5.593 s
gate = Recall/FDR/latency 全部通过
```

冒烟测试说明模型、tiled 推理、结果 schema 和时间记录均能正常工作；12 张样本不用于候选选择。

全量结果证据哈希：

| 产物 | SHA256 |
| --- | --- |
| raw tiled `result.json` | `822F87346A903F78CF6D53F9C0F0FA66A56482AC700AFA54330C37615723B92E` |
| raw tiled `timings.json` | `741178F405DD6E45C34B74AD8C7C36E2C21733FDF24F712F54918BA3EC4A265B` |
| raw tiled `official_metrics.json` | `F11063085EBE6EEBA0B0624B8F211BDDEAC6E0BB0B566957AD26C4209C84DBA6` |
| control `result.json` | `35C8BBCE55C0FF0BF3259788F7248DD364F92AD711A0A903AC21B1042048AD2D` |
| control `timings.json` | `1086896B4C08B4B675249A93E2D3BE8E89786C3ECB24FB5E2E6A599EC2288278` |
| control `official_metrics.json` | `B098A4C005122825149991CFEA2936CD219073856075D24EF2363D777ACD64D3` |
| gray027 `result.json` | `CF68576EC7EAA9618F9E54305C8894776CE1AB750A887F10C93BF98F6C7AE59A` |
| gray027 `timings.json` | `933EA9E204B13D0CA93A152594C74648DF12261563A122062025578DA3B5719A` |
| gray027 `official_metrics.json` | `9A5D9EC46CB25362EECD3A830312F8EE8421890DAF75D0AB01A87799A4C2A3FF` |

### 8.2 EXP004 原模型 tiled 全量

配置为 `test_inference.py`、`conf=0.30`、`iou=0.60`、tile `1024/0.20`、merge IoU `0.50`、tile batch `4`。

| 分组 | Recall | FDR | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| ship | 0.834601 | 0.174812 | 439 | 93 | 87 |
| aircraft | 0.983773 | 0.048417 | 3577 | 182 | 59 |
| vehicle | 0.802469 | 0.285714 | 65 | 26 | 16 |
| 三大类 pooled 宏平均 | 0.873614 | 0.169648 | - | - | - |

整体 TP/FP/FN 为 `4081/301/162`，总耗时 `34.024 s`，最大单图 `4.102 s`。历史兼容口径三项 gate 通过；按细分类别宏平均 Recall 为 `0.844286`，未达到 `0.85`，说明类别不均衡时结果仍有风险。

### 8.3 EXP004 control 与 gray027

两者均使用 `run_v2_modality_experiment.py` 的 tiled Detector、897 张相同验证图、相同模型和相同 FSC/飞机阈值，因而这组对照有效。

| 指标 | EXP004 control | EXP004 + gray027 | gray027 - control |
| --- | ---: | ---: | ---: |
| ship Recall | 0.834601 | 0.838403 | +0.003802 |
| ship FDR | 0.174812 | 0.189338 | +0.014526 |
| aircraft Recall | 0.983773 | 0.983773 | 0 |
| aircraft FDR | 0.048417 | 0.048417 | 0 |
| vehicle Recall | 0.777778 | 0.777778 | 0 |
| vehicle FDR | 0.222222 | 0.222222 | 0 |
| 三大类 pooled 宏 Recall | 0.865384 | 0.866651 | +0.001267 |
| 三大类 pooled 宏 FDR | 0.148484 | 0.153326 | +0.004842 |
| 最大单图耗时 | 3.531 s | 3.719 s | +0.188 s |

EXP004 control 的 TP/FP/FN 为 `4079/293/164`；gray027 为 `4081/303/162`。gray027 只多找回 2 个目标，同时增加 10 个误检，收益很小，control 的 FDR 余量更好。

### 8.4 与 EXP003 同策略对照

| 对照 | 宏 Recall 变化 | 宏 FDR 变化 | 结论 |
| --- | ---: | ---: | --- |
| EXP003 control -> EXP004 control | -0.017202 | -0.043320 | 召回下降，虚警明显下降 |
| EXP003 gray027 -> EXP004 gray027 | -0.016569 | -0.045445 | 召回下降，虚警明显下降 |

这说明继续训练没有带来整体召回提升，而是把模型推向更保守的预测：FDR 改善明显，但 Recall 下降。模型训练因素本身暂不能判定为正向提升。

## 9. 错误分析结论

EXP004 + `gray027` 错误分析覆盖 897 张图，其中 `235` 张存在至少一个错误，渲染了错误最多的 50 张图。

按误检/漏检数量，当前最需要关注：

| 类别 | 误检 FP | 漏检 FN | 判断 |
| --- | ---: | ---: | --- |
| MS | 75 | 55 | 船只组主要错误来源，既有重复/虚警也有漏检 |
| QHS | 27 | 26 | 类别边界和小目标识别不稳定 |
| FSC | 18 | 18 | vehicle 召回与虚警同时偏弱 |
| A13_F-15 | 27 | 4 | 典型飞机类别混淆，误检较集中 |
| A20_SU-24 | 1 | 11 | 漏检明显，应检查样本和标注 |

错误最多的图像同时出现重复飞机框、相近飞机类别混淆、MS/QHS 互相混淆和 FSC 漏检。当前问题更像类别/目标尺度/边界截断与数据分布问题，不能靠继续降低船只阈值解决。

## 10. 验收决定

- EXP004 模型和三套全量本地结果均已保留，结果 schema、覆盖率和推理时间正常。
- 按当前历史兼容口径，EXP004 raw、control、gray027 均通过三项 gate；按细分类别宏 Recall 口径，三者均低于 `0.85`，不能把本地结果写成稳妥通过。
- 在 EXP004 的同入口对照中，control 比 gray027 更稳：FDR 更低，召回只低 `0.001267`。当前暂选 control 作为 Docker 候选起点，gray027 作为备选，不直接提交。
- 继续训练的主要效果是降低 FDR、牺牲 Recall；不能据此认定模型训练已经改善隐藏测试表现。
- 下一步是把 EXP004 control 模型接入 Docker，做 `--network none` 的 12 张 smoke 和完整本地集成验收；通过后再讨论是否使用剩余平台提交次数。

## 报告图片素材

- [EXP004 训练曲线](../evidence/EXP004_original_continue40/results.png)
- [EXP004 验证集预测示例](../evidence/EXP004_original_continue40/val_batch0_pred.jpg)
- [EXP004 错误分析代表图](../evidence/EXP004_original_continue40/error_case_MAR20_193.jpg)
