# v1 Docker 运行依赖方案

状态：已用于 `submit/v1` Docker 构建并通过本地 GPU、断网全量验证；后续版本可另行调整

本文把研发环境和最终推理镜像的依赖边界分开。当前 `environment.yml` 已通过 `submit/v1` 的 Docker 构建和运行验证，但官方封闭测试结果仍待比赛平台返回。

## 一、依赖目标

v1 Docker 只服务于以下任务：

- 加载一个已经训练好的 YOLO 模型；
- 读取输入图片；
- 完成 direct 或 tiled 推理；
- 执行跨切片坐标还原和类别内 NMS；
- 生成官方要求的 result.json 和 timings.json。

因此，Docker 镜像不需要包含训练、数据审计、notebook、可视化和实验管理的完整依赖。

## 二、当前 Windows 环境中的已验证版本

以下版本来自 `weixing` 环境快照，用于解释研发环境与容器环境的对应关系；容器实际依赖以本目录的 `environment.yml` 为准：

| 组件 | 已验证版本 | 作用 |
| --- | --- | --- |
| Python | 3.10.20 | 运行时 |
| PyTorch | 2.5.1+cu121 | GPU 推理 |
| torchvision | 0.20.1+cu121 | PyTorch 视觉依赖 |
| Ultralytics | 8.4.128 | YOLO 模型加载与推理 |
| ultralytics-thop | 2.1.6 | Ultralytics 传递依赖 |
| NumPy | 1.26.4 | 数组与坐标处理 |
| OpenCV | 4.11.0.86 | 图片读取 |
| Pillow | 11.3.0 | 图像库传递依赖 |
| PyYAML | 6.0.3 | 配置读取传递依赖 |

当前 Windows 环境同时存在 opencv-python 和 opencv-python-headless。最终 Linux 镜像只选择 headless 版本，避免 GUI 相关依赖和两个 OpenCV 包互相覆盖。

## 三、交付入口的直接依赖

submit/v1/app 的代码直接使用：

- torch；
- ultralytics；
- cv2；
- numpy；
- pathlib、argparse、json、time 等 Python 标准库。

Ultralytics 还会带来若干必要的传递依赖，例如 Pillow、PyYAML、tqdm、psutil、scipy、pandas 和 requests。最终不应凭感觉删掉这些依赖，而应在 Linux 环境中实际创建并启动入口验证。

## 四、不放入 v1 运行镜像的内容

以下依赖属于研发或可选研究能力，当前不进入 v1 容器：

- pytest；
- JupyterLab、ipykernel；
- TensorBoard；
- matplotlib、seaborn；
- albumentations；
- pycocotools；
- jsonschema；
- scikit-learn；
- SAHI；
- ensemble-boxes；
- ONNX Runtime、TensorRT；
- notebook 和数据审计工具。

这些内容可以继续保留在 shiyan/environment/requirements.txt 中，服务于 Windows 研发和后续 v2 实验。

## 五、environment.yml 生成原则

1. 使用 Python 3.10。
2. 对齐官方 CUDA 12.1.1、Ubuntu 22.04 基础镜像。
3. PyTorch 和 torchvision 必须使用 CUDA 12.1 兼容版本。
4. 最终文件必须在 Linux x86_64 环境中创建并验证。
5. 不直接复制 Windows conda list。
6. 不把 win-64、pywin32、Windows 路径或 Windows 专用构建写进文件。
7. 先保证 app/main.py 可以离线启动，再考虑额外加速框架。
8. 运行时不下载模型、不下载依赖、不访问外部服务。

## 六、当前推荐的依赖层次

### 必要层

- Python 3.10；
- pip；
- PyTorch CUDA 12.1；
- torchvision；
- Ultralytics 8.4.128；
- OpenCV headless；
- NumPy；
- Ultralytics 的必要传递依赖。

### 暂不加入的层

- 训练增强和数据审计；
- COCO 评估工具；
- notebook 和绘图；
- ONNX、TensorRT；
- 其他检测框架。

这样做可以减小镜像体积，降低构建失败概率，也能让容器职责保持清晰。

## 七、与 Docker Desktop 的关系

当前不必安装 Ubuntu WSL。Docker Desktop 的 Linux 容器引擎可以提供临时 Linux x86_64 环境，用于后续环境文件创建和镜像验证。

Ubuntu WSL 的优势是有一个长期可操作的 Linux 终端，但它不是本项目 Docker 构建的硬性前置条件。无论选择哪条路线，最后都必须验证：

- environment.yml 能被官方 Dockerfile 中的 micromamba 读取；
- 镜像架构为 linux/amd64；
- 容器可以在 network none 下运行；
- GPU 可见且模型能够加载；
- result.json 满足官方协议。

## 八、当前验证状态

- [x] 通过 CUDA 12.1.1、Ubuntu 22.04 基础镜像构建 `linux/amd64` 镜像；
- [x] 在容器中安装 PyTorch CUDA 12.1、Ultralytics 8.4.128 和 OpenCV headless；
- [x] 使用 `submit/v1/app/main.py` 完成 GPU 离线启动测试；
- [x] 使用 `--network none` 完成 897 张全量推理；
- [x] 生成并核验 `result.json`、`timings.json` 和内部指标；
- [ ] 完成比赛 ACR 推送和官方封闭测试。

当前 `environment.yml` 是 `submit/v1` 的候选交付环境。若后续改变依赖，应创建新的版本目录并重新完成容器验证。
