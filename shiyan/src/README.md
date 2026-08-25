# src

后续代码模块位置。当前只建目录，不写实现。

建议边界：

- `data_audit/`：数据检查、哈希、类别统计。
- `dataset/`：数据读取、切片、格式转换。
- `modeling/`：模型构建和加载。
- `training/`：训练流程。
- `inference/`：推理入口和大图切片推理。
- `evaluation/`：官方指标本地复现。
- `postprocessing/`：NMS、WBF、阈值、异常框过滤。
- `packaging/`：提交包和 Docker 相关封装。
