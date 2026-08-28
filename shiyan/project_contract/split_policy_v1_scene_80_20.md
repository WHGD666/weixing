# Split Policy v1：scene_80_20

状态：frozen for first baseline，若后续修改划分策略，必须新建 v2，并说明旧实验不可直接比较。

## 划分目标

- 用于第一次 baseline 和后续可比实验。
- 目标比例：train:val = 8:2。
- 验证集用于内部模型选择和误差分析，不代表官方隐藏测试成绩。

## 原始数据保护

- `shiyan/data/` 保持官方原始状态。
- 不移动图片。
- 不移动标签。
- 不重命名文件。
- 不直接修改官方标签。
- 划分通过清单文件表达，而不是物理拆分目录。

## 划分单元

划分单元为 `scene_id`。当前 `scene_id` 由文件名中 `_cropN` 之前的部分推断。

示例：

```text
01-PAN-20240420-113-325-L00000010882-CCD3_5_crop4
```

对应：

```text
01-PAN-20240420-113-325-L00000010882-CCD3_5
```

同一个 `scene_id` 下的所有 crop 必须进入同一个 split，避免同一原始场景同时出现在训练集和验证集造成验证偏乐观。

## 类别约束

- train 和 val 都应覆盖 25 个细分类。
- 稀有类别 HM、LQS 必须保留验证样本，但不能被验证集拿走过多。
- 验证集的类别比例尽量接近全量数据比例。
- 后续报告应同时看 25 类细分指标和 ship / aircraft / vehicle 三大类指标。

## 固定参数

- split version：`v1_scene_80_20`
- input audit version：`v0_original`
- target val ratio：`0.20`
- seed：`20260828`
- search trials：`5000`

## 产物位置

```text
shiyan/data_registry/split_assignments/v1_scene_80_20/
├── train.txt
├── val.txt
├── image_assignments.csv
├── scene_assignments.csv
├── class_distribution_by_split.csv
├── group_distribution_by_split.csv
├── split_summary.csv
├── audit_checks.csv
├── metadata.json
├── split_fingerprint.txt
└── split_audit_report.md
```

YOLO 数据配置：

```text
shiyan/configs/dataset/weixing_v1_scene_80_20.yaml
```

## 使用原则

- 所有正式实验必须记录 split version 和 split fingerprint。
- 若人工修标生成新标签版本，应重新审计标签版本，并确认该 split 是否仍可复用。
- 若官方补发新数据，应先形成新的数据审计版本，再决定是否建立新的 split version。
