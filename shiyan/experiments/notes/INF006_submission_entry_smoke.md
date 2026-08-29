# INF006：submit/v1 交付入口本机冒烟测试

状态：completed，等待内部指标核验

## 目的

验证独立交付目录 `submit/v1/app` 的官方入口能够在当前 Windows + CUDA 环境中实际加载模型、使用 GPU、处理输入目录并写出官方 `result.json`。本测试不构建 Docker。

## 配置

- 入口：`submit/v1/app/main.py`
- 模型：`runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/best.pt`
- 输入：从 `v1_scene_80_20/val.txt` 取前 12 张图片复制到 `submit/v1/test-input/`
- 推理：tiled
- tile-size：1024
- tile-overlap：0.20
- merge-iou：0.50
- tile-batch：4
- conf：0.30
- NMS IoU：0.60

## 用户执行结果

- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- images：12
- total_seconds：5.790 秒
- max_image_seconds：5.384 秒
- result：`submit/v1/test-output/app_smoke_20260829/result.json`
- 入口内部 result schema 校验：通过

本次 12 张图片与 `INF004` 的均匀抽样样本不同，因此不能直接用耗时差异判断代码性能变化；需要先完成同口径指标核验。

## 清单问题记录

第一次调用 `evaluate_official.py` 时，手工生成的 `image_list.txt` 包含了测试目录中的非图片文件，导致评估器预期 14 条记录而结果文件只有 12 条。该失败属于测试清单生成错误，不属于交付入口或模型错误；后续清单必须按官方支持的四种图片后缀过滤，并放在临时输出目录中。

第二次评估使用了 `submit/v1/test-input` 下的图片路径。入口输出本身正确，但当前评估器会根据图片路径中的 `images` 目录自动推导对应 `labels` 目录，因此复制后的交付测试路径无法找到标签。后续本地指标评估使用原始 `shiyan/data/images/train` 路径生成清单；Docker 运行和官方评测不读取标签，不受此限制。

修正清单后，12 张样本的评估任务能够正常完成，FDR 和延迟门槛通过，Recall 门槛未通过。该 Recall 标志只反映前 12 张小样本，不能替代 897 张固定验证集结果，也不作为 Docker 候选淘汰依据；本次测试的主要结论仍是交付入口的 GPU、输出格式和文件覆盖率正常。

## 结论

交付入口已经具备继续评估条件：GPU 检查、模型加载、第一层图片读取、tiled 推理、结果写出和 schema 校验均已走通。当前不复制 Dockerfile、不生成 `environment.yml`、不构建镜像、不推送 Git。
