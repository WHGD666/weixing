# submissions

提交管理目录。

目录：

- `candidates/`：候选提交版本。
- `frozen/`：冻结后的提交版本记录。
- `official_feedback/`：官方反馈记录，例如 `SUB001_trial_2551.md`。

当前已完成两次官方试运行和三次正式提交。正式结果统一见 [`../experiments/PROJECT_STATUS_20260904.md`](../experiments/PROJECT_STATUS_20260904.md) 与 [`../experiments/registry/submission_registry.csv`](../experiments/registry/submission_registry.csv)。三次正式提交均成功执行，但按三大类算术平均口径均未同时通过 Recall/FDR 门槛；`FORMAL001` 仍是正式综合分最高版本，`FORMAL003` 的 Recall 更接近门槛但 FDR 更差。

正式复盘：[`FORMAL001`](official_feedback/FORMAL001_v1_3496.md)、[`FORMAL002`](official_feedback/FORMAL002_v2_3720.md)、[`FORMAL003`](official_feedback/FORMAL003_v3_3872.md)。

注意：不要把内部验证结果写成官方隐藏测试成绩。
