# src

## 当前内容

`inference/` 是本项目的可复用推理模块，训练和 Docker 交付都应尽量复用同一套逻辑：

- `labels.py`：冻结的 25 类顺序和大类映射。
- `types.py`：统一的像素坐标检测框类型。
- `predictor.py`：Ultralytics 模型加载、整图推理和大图切片推理。
- `postprocess.py`：跨切片的按类别 NMS 和 IoU 计算。
- `runner.py`：输入目录读取、官方 `result.json` 生成、可视化和耗时旁路记录。
- `schema.py`：对官方结果字段、类别、置信度和像素坐标框进行严格校验。

官方 Docker 的 `app/main.py` 还没有在当前阶段创建；最终交付时应调用这里的 runner，避免 Docker 和本地测试出现两套推理逻辑。

## 重要约束

- 结果框统一为原图像素坐标 `[x1, y1, x2, y2]`。
- `category_id` 必须与 `category_name` 和项目标签契约一致。
- 官方输入目录只读取第一层的 `.jpg`、`.jpeg`、`.png`、`.bmp` 文件。
- `timings.json` 只用于本地耗时分析，不替代官方 `result.json`。
- 本模块不负责训练，也不在运行时下载模型。

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
