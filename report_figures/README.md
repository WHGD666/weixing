# YOLO 实验报告图片包

本目录用于小组实验报告，所有图片均由当前仓库中的真实实验产物生成或复制，未手工编造数据。总大小约 3.8 MB，适合直接拷贝到 U 盘或随 GitHub 仓库下载。

## 目录说明

### `01_detection_examples`

带真实检测框的样例图，可用于报告中的“可视化检测结果”或“定性分析”。

- `EXP001_train_labels.jpg`：EXP001 原始基线训练批次，展示数据标签。
- `EXP002_val_predictions.jpg`：EXP002 人工修订数据训练后的验证预测。
- `EXP004_val_predictions.jpg`：EXP004 原始标签继续训练 40 轮后的验证预测。
- `EXP004_error_case_MAR20_193.jpg`：EXP004 gray027 后处理实验中的错误样例，用于说明漏检/误检分析。

### `02_training_figures`

可用于“训练过程”和“模型诊断”部分。

- `EXP001_training_curves.png`：EXP001 基线训练曲线。
- `EXP002_training_results.png`、`EXP004_training_results.png`：Ultralytics 原生训练结果图。
- `EXP002_confusion_matrix_normalized.png`、`EXP004_confusion_matrix_normalized.png`：归一化混淆矩阵。

### `03_metric_figures`

可用于“实验对比”和“指标结果”部分。

- `training_metrics_comparison.png`：三次训练的验证 Recall、mAP50 随 epoch 变化。
- `local_validation_gate_comparison.png`：EXP003/EXP004 本地验证的 Recall、FDR 和最大单图耗时对比，含 0.85、0.20、20 s 门槛线。
- `official_hidden_test_comparison.png`：正式提交 `v1.0`（提交 3496）与 `v2.0`（提交 3720）的平台回传 Recall/FDR 对比。
- `EXP003_modality_metrics.png`：颜色/灰度后处理策略的指标对比。

### `04_error_and_data_figures`

可用于“数据集分析”和“误差分析”部分。

- `EXP004_control_class_metrics.png`：EXP004 control 本地验证的 25 类 Recall/FDR。
- `EXP004_error_analysis_FP_FN.png`：EXP004 gray027 中 FP+FN 最大的前 12 类。
- `dataset_object_distribution.png`：固定 `v1_scene_80_20` 划分下各类别训练/验证目标数量。
- `validation_modality_split.png`：897 张验证图中的灰度和彩色图数量。

## 引用边界

- `official_hidden_test_comparison.png` 的数值来自比赛平台提交记录，是正式隐藏测试回传结果。
- 其余指标图中的 `EXP00x` 结果属于本地验证，只能写成“内部验证结果”，不能写成官方隐藏测试成绩。
- 本地门槛图的 Recall/FDR 使用项目评估器的 group-mean 口径；报告引用时应同时写明指标口径。
- 训练曲线中的 Recall/mAP50 是 Ultralytics 验证集指标，与比赛平台的三大类宏平均指标不是同一指标。

## 数据源与复现

图表由 `shiyan/scripts/generate_report_figures.py` 生成，运行环境使用现有 `weixing` Conda 环境中的 Python 3.10.9 和 matplotlib 3.10.9。用户指定的 `D:\daima\shumochuangxinbei\.venv` 当前不存在，因此没有向该路径安装或修改任何内容。

主要数据源：

- `runs/detect/runs/train/*/results.csv`
- `runs/test/EXP003_v2_modality_*/metrics/official_metrics.json`
- `runs/test/EXP004_*/metrics/official_metrics.json`
- `runs/test/EXP004_modality_gray027/error_analysis/error_per_class.csv`
- `shiyan/data_registry/split_assignments/v1_scene_80_20/class_distribution_by_split.csv`
- `shiyan/experiments/registry/submission_registry.csv`

推荐报告组合：每个实验选 1 张带框图 + 1 张训练/指标图；总报告不必全部插入，可按章节挑选。
