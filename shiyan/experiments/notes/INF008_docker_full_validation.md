# INF008：Docker 交付镜像全量离线验证

状态：completed，Docker 本地全量验证通过，等待比赛平台试运行

## 1. 实验定位

本记录验证 `submit/v1` 的实际 Docker 镜像能否在接近比赛评测的条件下完成完整推理。它不是新的模型实验，也不改变 `EXP001` 的模型、数据、标签、划分或推理参数。

本次验证重点检查：

- 镜像架构是否为 `linux/amd64`；
- GPU 是否能被容器识别；
- 断网后是否仍能运行；
- 入口是否只加载一次模型并处理全部输入图片；
- `/output/result.json` 和 `timings.json` 是否正常生成；
- 输出是否能通过项目内部官方指标核验。

## 2. 固定配置

| 项目 | 值 |
| --- | --- |
| 交付版本 | `submit/v1` |
| 本地镜像 | `weixing-submission:v1` |
| 镜像架构 | `linux/amd64` |
| 基础镜像 | `nvidia/cuda:12.1.1-base-ubuntu22.04` |
| 输入 | `submit/v1/test-input/`，897 张固定验证图片 |
| 网络 | `--network none` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 推理模式 | tiled |
| tile-size | 1024 |
| tile-overlap | 0.20 |
| merge-iou | 0.50 |
| tile-batch | 4 |
| conf | 0.30 |
| NMS IoU | 0.60 |
| 模型 SHA-256 | `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |

## 3. 实际运行结果

| 项目 | 结果 |
| --- | --- |
| 处理图片数 | 897 |
| 预测结果 | `submit/v1/test-output/docker_full_20260829/result.json` |
| 耗时记录 | `submit/v1/test-output/docker_full_20260829/timings.json` |
| 总耗时 | 50.588104 秒 |
| 最大单图耗时 | 1.800521 秒 |
| 结果图片覆盖 | 897/897 |
| GPU 检查 | 通过 |
| 断网运行 | 通过 |
| Docker 镜像 inspect 大小 | 3,922,192,555 字节 |
| Docker 镜像 ID | `414cb1e6e2a8` |

容器启动时出现的 Ultralytics 配置目录提示只是因为 `/root/.config/Ultralytics` 不可写，程序自动切换到 `/tmp/Ultralytics`，不影响模型加载、推理结果或断网运行。本次不为消除该提示重新构建镜像。

## 4. 内部官方口径核验

| 指标 | 结果 | 门槛 | 判断 |
| --- | ---: | ---: | --- |
| Overall Recall | 0.9655903842 | >= 0.85 | 通过 |
| Overall FDR | 0.0709750567 | <= 0.20 | 通过 |
| Max image time | 1.800521 秒 | <= 20 秒 | 通过 |

总体 TP / FP / FN 为 `4097 / 313 / 146`。评估脚本返回：

```json
{"recall_ge_0_85": true, "fdr_le_0_20": true, "latency_le_20s": true}
```

完整指标文件：

```text
submit/v1/test-output/docker_full_20260829/metrics/official_metrics.md
```

以上是公开验证集上的内部核验结果，不是比赛封闭测试集的官方成绩。官方测试仍需由比赛系统拉取镜像后重新评测。

## 5. 结果解释

- Docker 入口与本机 `submit/v1/app/main.py` 的结果一致，说明模型路径、类别表、切片坐标还原、后处理和结果结构已经接通。
- `--network none` 下仍可运行，说明镜像没有依赖启动时联网下载模型或依赖。
- 当前最大单图耗时明显低于 20 秒，具备提交试运行的工程余量。
- 本地验证集的图像尺寸和官方封闭测试集不一定完全相同，因此不能把本地指标直接写成官方成绩。
- 舰船和发射车分组仍是风险点；本次先不人工修标、不继续扩大参数扫描，保留为 v2 的后续研究方向。

## 6. 阶段性冻结决策

`submit/v1` 进入“本地交付候选已验证、官方试运行结果已归档”的阶段性冻结：

- 冻结原始标签版本 `v0_original`；
- 冻结验证协议 `v1_scene_80_20`；
- 冻结模型及其 SHA-256；
- 冻结 tiled 推理参数和官方输出接口；
- 官方提交 `2551` 的结果单独记录在 `shiyan/submissions/official_feedback/SUB001_trial_2551.md`；不冻结后续人工审计方向。

任何模型、标签、阈值、切片策略或入口代码的实质变化，都应创建 `submit/v2`，不得直接覆盖本候选。

## 7. 后续动作

1. 保留官方提交 `2551` 作为第一轮原始标签基线。
2. 根据官方分组指标和既有错误分析，决定是否启用 `AUDIT001`。
3. 如需改标、重新训练或改动推理参数，创建 `EXP002`/`submit/v2`，不得覆盖 v1。
