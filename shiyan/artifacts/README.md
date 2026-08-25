# artifacts

实验产物证据区。大文件和可再生成产物默认不直接进入 Git，重要小表格、manifest、汇总结果可以进入 Git。

目录：

- `tables/`：指标表、混淆矩阵、类别统计。
- `predictions/`：验证集、holdout 或官方测试预测文件。
- `event_logs/`：run event、日志摘要、错误摘要。
- `figures/`：可视化图和报告图。
- `reports/`：自动生成报告。
- `frozen_model_artifacts/`：冻结模型的清单和哈希，权重本体通常不进 Git。
- `submission_packages/`：提交包清单和哈希，包体通常不进 Git。
