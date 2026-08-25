# 环境方案说明

本目录只记录比赛项目需要的 Python 依赖清单、安装顺序和环境冻结方法，不存放 conda 环境、第三方库源码、模型权重或数据集。

当前本机环境状态：

- conda 环境名：`weixing`
- Python：`3.10.20`
- 当前包状态：只有 Python、pip、setuptools、wheel 和系统基础运行库
- 本机训练卡：NVIDIA GeForce RTX 4060 Laptop GPU，8GB 独显显存

这个状态是一个合适的起点。后续环境应按阶段安装，先保证 baseline 可跑通，再补充大图推理、融合和研究型框架。

## 推荐安装顺序

第一步，安装 PyTorch CUDA 版本：

```powershell
python -m pip install -r shiyan/environment/requirements-torch-cu121.txt
```

第二步，安装主线训练、数据处理和 COCO 输出依赖：

```powershell
python -m pip install -r shiyan/environment/requirements-main.txt
```

第三步，安装实验记录、测试和 notebook 辅助依赖：

```powershell
python -m pip install -r shiyan/environment/requirements-dev.txt
```

第四步，等 baseline 跑通后再安装大图推理和融合依赖：

```powershell
python -m pip install -r shiyan/environment/requirements-inference.txt
```

## 安装后必须验证

安装后至少执行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import cv2, numpy, pandas, pycocotools; print('basic imports ok')"
```

如果 `torch.cuda.is_available()` 为 `False`，不要继续训练，先处理 PyTorch CUDA 版本、显卡驱动或环境路径问题。

## 环境冻结

第一次完整安装并验证通过后，保存实际版本：

```powershell
python -m pip freeze > shiyan/environment/locks/weixing-pip-freeze.txt
conda list > shiyan/environment/locks/weixing-conda-list.txt
```

正式实验记录中必须写清楚使用的是哪一份 lock 文件。未来如果升级库版本，应该新增 lock 文件或更新环境说明，并说明旧实验和新实验是否还能直接比较。

## 暂缓安装

以下内容先不要装进主环境：

- `mmdetection` / `mmcv` / `mmengine`：研究价值高，但 Windows + CUDA + PyTorch 版本绑定复杂，适合 baseline 稳定后单独开研究环境。
- `detectron2`：Windows 原生安装成本高，不适合作为第一主线。
- `tensorrt`：等模型和提交接口稳定后再做推理加速。
- `wandb`：如果后续需要线上实验看板再启用，当前阶段本地 CSV/JSON 记录更稳。

## 本机训练约束

RTX 4060 Laptop 8GB 适合做 baseline、消融实验和小规模调参，不适合直接按最终 RTX3090 的吞吐量设计训练流程。

建议默认策略：

- 使用小模型或中小模型起步。
- 开启 AMP 混合精度。
- batch size 从小值开始，遇到 OOM 再降。
- 大图训练和推理走切片，不做 10000x10000 整图硬推。
- 优先把数据审计、划分冻结、指标计算和提交格式跑通，再追求大模型。
