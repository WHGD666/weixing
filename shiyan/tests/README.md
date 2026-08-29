# tests

这里放不依赖真实数据和模型权重的单元测试，以及对官方输出契约的固定检查。

当前测试覆盖：

- 同类别重叠框会被 NMS 抑制，不同类别不会互相抑制。
- 官方 `result.json` 的状态、图片覆盖、类别映射、置信度、坐标范围和框格式。
- 越界框会被拒绝。

测试命令由项目成员在本机执行：

```powershell
python -m pytest shiyan/tests/test_inference_contract.py -q
```

真实模型推理请使用 `shiyan/scripts/test_inference.py`，不要把单元测试结果当成模型效果结果。

测试目录。正式训练前至少要有数据契约和指标契约测试。

目录：

- `data_contract/`：类别 ID、坐标范围、split 互斥等检查。
- `metrics/`：TP/FP/FN、IoU、Recall、FDR 的小样例测试。
- `smoke/`：环境、推理入口、输出格式冒烟测试。
