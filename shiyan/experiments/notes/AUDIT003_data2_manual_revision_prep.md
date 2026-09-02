# AUDIT003：data2 第一轮人工修订入训准备

状态：**frozen_as_exp002_candidate，训练和平台提交已完成**

本记录对应 2026-09-01 的第一轮人工标签修订。用户已完成前 1360 张图片的人工检查和微调，主要关注舰船与 FSC/导弹车；没有进行大规模新增标注。原始 `shiyan/data` 和 `v0_original` 保持只读，`shiyan/data2` 是人工修改工作副本。该候选已完成训练、Docker 本地验收和正式平台提交，结果作为 `EXP002` 冻结，后续不覆盖本实验材料。

## 1. 当前范围

| 项目 | 当前值 |
| --- | --- |
| 图片目录 | `shiyan/data2/images/train` |
| 标签目录 | `shiyan/data2/labels/train` |
| 图片数量 | `4481` |
| 当前标签文件数量 | `4481` 个同名标签文件（含 4 个空标签） + `classes.txt` |
| 人工修订范围 | `0001` 至 `1360` |
| 未修改范围 | `1361` 至 `4481` |
| 类别数 | `25` |
| 原始标签版本 | `v0_original` |
| 候选标签版本 | `v1_manual_revision_candidate` |
| 训练状态 | 已完成 55 轮；候选已冻结 |

## 2. 已发现的文件级问题

编号 `0581`、`0582`、`0583`、`0910` 经用户确认是没有可用目标信息的空白标注图。当前它们没有对应 `.txt`，语义上应当是空标签，但文件级上仍需创建 4 个空的同名标签文件，才能让图片和标签严格一一对应。

这 4 个文件不应添加任何检测框。只创建空文件，不要填入 `0 0 0 0 0`，后者会制造一个错误的 HM 目标。

## 3. 与原始标签的静态对照

前 1360 张的原始文本几乎全部被标注工具重新保存，因此不能用文本 diff 判断人工修改数量。按类别行数对照，当前候选版本的净变化为：

| 类别 | 原始目标行数 | data2 目标行数 | 净变化 | 解释 |
| --- | ---: | ---: | ---: | --- |
| QHS | 641 | 638 | -3 | 有删框或类别调整 |
| MS | 1994 | 1982 | -12 | 有删框或类别调整 |
| FSC | 402 | 399 | -3 | 有删框/误标清理 |
| 其他类别合计 | 不变 | 不变 | 0 | 未观察到总量变化 |
| 全部类别 |  |  | -18 | 没有大规模新增目标 |

进一步的类别计数差异出现在 92 张图，主要集中于 QHS/MS/FSC。坐标微调需要在空标签补齐后通过机器格式检查，并结合可视化抽查确认，不能仅凭保存后的文本差异下结论。

## 4. 入训前必须执行的命令

以下命令由项目成员在 `D:\daima\weixing`、已激活的 `(weixing)` 环境中执行。Codex 不代替执行标签验收或训练。

### 4.1 补齐四个空标签

```powershell
cd D:\daima\weixing
conda activate weixing

$labelRoot = "D:\daima\weixing\shiyan\data2\labels\train"
foreach ($stem in @("0581", "0582", "0583", "0910")) {
  $path = Join-Path $labelRoot "$stem.txt"
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType File -Path $path | Out-Null
  }
}
```

### 4.2 YOLO 标签机器验收

```powershell
python shiyan/scripts/check_data2_yolo_labels.py --root shiyan/data2
```

预期输出：

```text
ok=true images=4481 labels=4481 classes=25
```

只要不是这条结果，就先不要训练，把完整报错贴回来。

### 4.3 生成与原始场景划分一致的 data2 清单

```powershell
python shiyan/scripts/prepare_data2_split.py
```

实际输出为 `image_count=4481`，训练集 `3584` 张，验证集 `897` 张。该脚本只生成基于固定 `v1_scene_80_20` 的新文件清单，不改变图片和标签。

### 4.4 检查训练配置

```powershell
Get-Content shiyan/configs/dataset/weixing_data2_v1_audited.yaml
Get-Content shiyan/configs/train/exp002_data2_manual_revision.yaml
```

## 5. 训练命令

完成 4.1 至 4.4 且验收通过后，再由项目成员启动第一轮训练。已按项目成员要求将训练上限调整为 55 轮：

```powershell
yolo detect train cfg=shiyan/configs/train/exp002_data2_manual_revision.yaml
```

配置文件虽然写入 `model: submit/v2/models/best.pt`，但实际运行日志和配置包含 `pretrained: false`。当前 Ultralytics 会在这种设置下丢弃 checkpoint 权重，因此本次实际是从头初始化训练，不是基于旧模型微调。这是本次实验的重要混杂因素。输入尺寸 `1024`、batch `4`、GPU `0`、最多 55 轮。若 RTX 4060 Laptop 显存不足，只把命令改为：

```powershell
yolo detect train cfg=shiyan/configs/train/exp002_data2_manual_revision.yaml batch=2
```

不要改变数据清单、类别顺序或随机种子来规避显存问题。

## 6. 训练完成后的回传结果

训练已完成，模型为 `runs/detect/runs/train/exp002_data2_manual_revision/weights/best.pt`，SHA256 为 `9378660F53BF06E613B580188D724D58647B3E1BF02A7752AB8A0FA77963D8CD`。第 55 轮日志为 Precision `0.91108`、Recall `0.79155`、mAP50 `0.88510`、mAP50-95 `0.66950`。同一 `v1_scene_80_20_data2` 验证清单上的本地官方口径复现和 Docker smoke 均已完成，详细结果见 [EXP002_data2_manual_revision.md](EXP002_data2_manual_revision.md)；平台结果见 [FORMAL002_v2_3720.md](../../submissions/official_feedback/FORMAL002_v2_3720.md)。

## 7. 冻结判断

这轮修改的方向与正式提交暴露的问题一致：集中在舰船和 FSC，而不是盲目改动全部类别；从类别总量看没有大量新增框，主要是删框、类别调整和框微调。4 个空标签已补齐，机器验收和固定划分映射均已通过。但平台正式结果 `v2.0` 得分 `58.0075`，低于 `FORMAL001` 的 `70.5805`，且三类宏平均门槛未通过。因此 `EXP002` 作为失败但可追溯的冻结实验保留，`v1.0` 继续作为当前正式基线；下一轮必须先修正 `pretrained: false` 的配置混杂，再单独评估标签修订效果。
