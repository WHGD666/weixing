# 实验报告图片素材

这里保存少量可直接用于实验报告的图片。图片均来自已完成的本地训练、验证或错误分析产物，未上传模型权重、原始数据和完整可视化目录。

> 所有指标图和检测图都是内部验证材料，不是比赛隐藏测试结果。

## 素材目录

### EXP001：原始标签 baseline

- [training_curves.png](EXP001_original_baseline/training_curves.png)：原始标签 baseline 的训练损失和内部验证指标曲线。
- [train_batch0.jpg](EXP001_original_baseline/train_batch0.jpg)：原始训练样本及标签示例。

### EXP002：data2 人工修订标签

- [results.png](EXP002_data2_manual_revision/results.png)：人工修订标签模型的训练曲线。
- [val_batch0_pred.jpg](EXP002_data2_manual_revision/val_batch0_pred.jpg)：验证集预测可视化示例。

### EXP003：图像模态辅助后处理

- [modality_metrics.png](EXP003_modality_postprocess/modality_metrics.png)：control、`gray027` 和低阈值 soft 的宏 Recall/FDR 内部对比；虚线为当前 gate。

### EXP004：原始标签权重继续训练 40 轮

- [results.png](EXP004_original_continue40/results.png)：继续训练 40 轮的训练曲线。
- [val_batch0_pred.jpg](EXP004_original_continue40/val_batch0_pred.jpg)：验证集预测可视化示例。
- [error_case_MAR20_193.jpg](EXP004_original_continue40/error_case_MAR20_193.jpg)：错误分析代表图，展示典型漏检/类别混淆问题。

## 使用说明

报告中建议每个实验选 1 到 2 张图即可：训练类实验使用训练曲线和预测示例，后处理实验使用指标对比图，错误分析部分使用 EXP004 错误案例图。不要把这些图片当作官方平台结果截图。
