# v1 交付阶段性冻结清单

状态：**official-trial-completed，v1 官方试运行结果已归档**

本文记录第一轮原始数据基线的交付对象。它是 `v1` 的版本边界，不替代比赛官方最终评测结果。

## 一、候选身份

| 项目 | 当前值 |
| --- | --- |
| 交付版本 | `submit/v1` |
| 数据标签 | 比赛方原始标签，`v0_original` |
| 数据划分 | `v1_scene_80_20` |
| 训练实验 | `EXP001_original_yolo11s_baseline` |
| 模型路径 | `runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/best.pt` |
| 本地构建模型 | `submit/v1/models/best.pt` |
| 模型大小 | 76,124,886 字节 |
| 模型 SHA-256 | `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |
| 本地镜像 | `weixing-submission:v1` |
| 镜像架构 | `linux/amd64` |
| 目标设备 | CUDA GPU |
| 当前状态 | 已完成 ACR 推送和官方试运行，提交 `2551` 已计分 |

## 二、冻结的推理参数

| 参数 | 值 |
| --- | --- |
| imgsz | 1024 |
| conf | 0.30 |
| NMS IoU | 0.60 |
| tile-size | 1024 |
| tile-overlap | 0.20 |
| merge-iou | 0.50 |
| tile-batch | 4 |
| max-det | 300 |

除非创建新的 `submit/v2`，不应在 `v1` 中临时改变这些参数。

## 三、已经验证的内容

### Windows 入口

- 入口：`submit/v1/app/main.py`；
- 输入：897 张固定验证图片；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU；
- 总耗时：40.793 秒；
- 最大单图耗时：5.049 秒；
- Overall Recall：0.965590；
- Overall FDR：0.070975；
- 结果结构：通过内部 schema 校验；
- 内部门槛判断：Recall、FDR、Latency 均为 true。

### Docker 入口

- 镜像：`weixing-submission:v1`；
- 架构：`linux/amd64`；
- 网络：`--network none`；
- 输入：897 张固定验证图片；
- GPU 检查：通过；
- 总耗时：50.588104 秒；
- 最大单图耗时：1.800521 秒；
- Overall TP / FP / FN：4097 / 313 / 146；
- Overall Recall：0.9655903842；
- Overall FDR：0.0709750567；
- 内部门槛判断：Recall、FDR、Latency 均为 true。

完整证据见：

```text
submit/v1/test-output/docker_full_20260829/
```

以上是公开验证集上的内部结果，不是官方封闭测试成绩。详细运行记录见 [`INF008_docker_full_validation.md`](../../../shiyan/experiments/notes/INF008_docker_full_validation.md)。

## 四、当前已冻结的边界

- 原始数据保持只读；
- 标签版本固定为 `v0_original`；
- 验证协议固定为 `v1_scene_80_20`；
- 25 类类别顺序和输出 schema 固定；
- 模型文件及 SHA-256 固定；
- tiled 推理参数固定；
- `/input`、`/output` 和 `/output/result.json` 接口固定；
- 本地 Docker 构建和断网全量验证证据保留。

## 五、官方试运行结果

- [x] 使用平台临时凭据完成 ACR 推送；
- [x] 推送 `trial-v1.0` 镜像，manifest digest 为 `sha256:414cb1e6e2a8d419d0f55ba3a6137fae15dc95bf84c523e449e589dbfc1c3d99`；
- [x] 提交官方试运行，提交 ID 为 `2551`；
- [x] 平台状态为 `ACCEPTED`，任务已完成且计入成绩；
- [x] 综合分 `84.3313`，排名第 42；
- [x] 官方页面可见指标已记录到 [`SUB001_trial_2551.md`](../../../shiyan/submissions/official_feedback/SUB001_trial_2551.md)。

官方页面未展示整体 Recall、整体虚警率的完整字段，不能用分组字段反推整体值；本地公开验证集指标与官方封闭测试结果仍须分开书写。

## 六、交付文件边界

应进入 GitHub：

- `app/` 推理代码；
- `Dockerfile`；
- `environment.yml`；
- manifest、实验记录、数据协议、测试代码和说明文档。

不进入 GitHub：

- 训练数据、标签和测试图片；
- `models/best.pt`；
- `test-input/` 和 `test-output/` 的本地内容；
- `runs/` 训练和推理缓存；
- 代理配置、密码和 ACR 临时凭证。

## 七、后续版本规则

如果后续使用人工审计标签、改动模型、改动切片策略、改动阈值或改动交付入口，应建立 `submit/v2`。`v1` 保持原始数据基线，不与 `v2` 共享可变模型和交付文件。
