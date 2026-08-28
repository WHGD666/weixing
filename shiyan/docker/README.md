# docker

Docker 封装与评测系统资料目录。

当前已放入比赛方提供的 Dockerfile、Docker 封装说明、评测系统使用手册和答疑记录。现阶段这些文件主要作为交付接口依据；正式推理代码、模型权重和 Linux `environment.yml` 等到 baseline 和推理流程稳定后再整理。

重点记录：

- 基础镜像与 CUDA/PyTorch 版本。
- `environment.yml` 的 Linux x86_64 生成要求。
- `/input` 与 `/output/result.json` 的官方接口。
- 推理入口和模型权重位置。
- 本地 Docker 冒烟测试命令。
- ACR 镜像 tag、push 和评测提交流程。
