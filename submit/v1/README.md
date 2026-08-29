# submit/v1：第一轮原始数据基线交付候选

状态：**官方试运行已完成，v1 基线结果已归档**

这是第一轮原始数据基线的独立交付工作区。它使用比赛方原始标签，不使用人工审计修订标签；后续改进必须创建 `submit/v2/`，不得覆盖本版本。

## 当前候选

| 项目 | 值 |
| --- | --- |
| 训练实验 | `EXP001_original_yolo11s_baseline` |
| 标签版本 | `v0_original` |
| 验证划分 | `v1_scene_80_20` |
| 模型 | `models/best.pt`，仅保留在本地构建上下文 |
| 推理模式 | tiled |
| tile-size / overlap | 1024 / 0.20 |
| merge-iou | 0.50 |
| conf / NMS IoU | 0.30 / 0.60 |
| 模型 SHA-256 | `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |
| 本地 Docker 验证 | 897 张，`--network none`，GPU 通过 |
| 本地内部 Recall / FDR | `0.965590` / `0.070975` |
| Docker 最大单图耗时 | `1.800521s` |
| 官方封闭测试 | 提交 `2551`，`ACCEPTED`，综合分 `84.3313`，排名第 42 |

## 目录职责

```text
submit/v1/
├── README.md
├── Dockerfile                 # linux/amd64 构建文件
├── environment.yml            # 容器运行环境
├── app/                       # 入口、检测器、标签表、后处理和 schema
├── models/                    # 本地模型构建上下文，不进入 Git
├── manifests/                 # 版本边界、依赖和验证记录
├── test-input/                # 本地测试图片，不进入 Git
└── test-output/               # 本地运行结果，不进入 Git
```

## 已完成验证

- `python -m py_compile` 通过；
- `main.py --help` 通过；
- Windows GPU 入口完成 897 张全量运行；
- Docker 镜像架构为 `linux/amd64`；
- Docker 在 GPU 和 `--network none` 下完成 897 张全量运行；
- 输出完整覆盖输入图片，内部结果 schema 和官方口径评估通过；
- 详细 Docker 记录见 [`INF008_docker_full_validation.md`](../../shiyan/experiments/notes/INF008_docker_full_validation.md)。
- 官方结果记录见 [`SUB001_trial_2551.md`](../../shiyan/submissions/official_feedback/SUB001_trial_2551.md)。

## 交付边界

进入 Docker 构建上下文的内容：

- `Dockerfile`；
- `environment.yml`；
- `app/`；
- 本地 `models/best.pt`。

不进入 GitHub 源码仓库的内容：

- 训练数据和标签；
- `models/best.pt` 权重；
- `test-input/` 图片；
- `test-output/` 结果；
- `runs/` 训练和推理缓存；
- Windows 绝对路径、代理配置和登录凭据。

## 当前状态

`trial-v1.0` 已完成 ACR 推送和官方试运行，平台显示提交 `2551` 为 `ACCEPTED`、任务已完成且计入成绩。`v1` 作为原始标签对照基线保留；后续人工审计、改标、重新训练或参数变化统一进入 `submit/v2`。

本地内部指标只用于研发记录，不等同于官方封闭测试成绩。
