# EXP004：原始标签模型权重继续训练 40 轮

状态：**training_completed，待完成官方口径推理验收；未提交平台**

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
