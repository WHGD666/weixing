# docker

Docker 封装与评测系统资料目录。

当前已放入比赛方提供的 Dockerfile、Docker 封装说明、评测系统使用手册和答疑记录。`submit/v1` 已完成推理代码、Linux `environment.yml`、模型构建上下文和 Docker 本地全量验证；本目录继续作为官方约束和评测流程的资料区。

重点记录：

- 基础镜像与 CUDA/PyTorch 版本。
- `environment.yml` 的 Linux x86_64 生成要求。
- `/input` 与 `/output/result.json` 的官方接口。
- 推理入口和模型权重位置。
- 本地 Docker 冒烟测试命令。
- ACR 镜像 tag、push 和评测提交流程。

当前准备策略：

- Windows + Conda 继续负责训练和实验；
- Docker Desktop 的 Linux 容器引擎负责后续镜像构建与运行验证；
- 不强制额外安装 Ubuntu WSL；
- 最终 environment.yml 仍必须对应 Linux x86_64，并通过容器实际验证。

当前 Docker 状态：

- 镜像：`weixing-submission:v1`；
- 架构：`linux/amd64`；
- GPU + `--network none` 全量处理 897 张图片；
- 内部 Recall/FDR/Latency 三项门槛均通过；
- ACR push 已完成；官方提交 `2551` 已 `ACCEPTED`、任务已完成，综合分 `84.3313`，排名第 42。详细结果见 `shiyan/submissions/official_feedback/SUB001_trial_2551.md`。
