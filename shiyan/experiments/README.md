# experiments

多轮实验管理目录。这里记录实验事实，不只记录成功结果。

目录：

- `registry/`：实验总表和 run 总表。
- `run_manifests/`：每次正式 run 的不可变 manifest。
- `comparisons/`：同一协议下可比较实验的对比表。
- `failures/`：失败实验、异常、拒绝原因。
- `notes/`：阶段性实验复盘。

## run_id 建议

```text
YYYYMMDD_HHMM_task_experiment_shortid
```

示例：

```text
20260826_0100_det25_baseline_yolo_v0
```
