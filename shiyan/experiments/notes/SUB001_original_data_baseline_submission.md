# SUB001：原始比赛数据基线提交候选

状态：**official_result_received，第一轮官方试运行已归档**

## 1. 实验定位

在不进行人工修标、不引入外部数据、不改变冻结划分的前提下，把第一版模型封装为可运行 Docker 候选并提交，获得一个真实官方评测基准。当前已完成本地验证、ACR 推送和比赛系统官方试运行。

本候选的作用是建立第一条可靠对照线：以后人工审计、模型改进、阈值变化或推理优化，都必须与本版本区分，不能把不同标签和不同交付配置混在一起比较。

## 2. 候选配置

| 项目 | 值 |
| --- | --- |
| 训练实验 | `EXP001_original_yolo11s_baseline` |
| 模型 | `runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/best.pt` |
| 交付模型 | `submit/v1/models/best.pt` |
| 标签 | 比赛方原始标签，`v0_original` |
| Split | `v1_scene_80_20` |
| 推理模式 | tiled |
| tile-size / overlap | 1024 / 0.20 |
| merge-iou | 0.50 |
| conf / NMS IoU | 0.30 / 0.60 |
| max-det | 300 |
| 模型 SHA-256 | `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |
| 官方封闭测试 | 提交 `2551`，`ACCEPTED`，综合分 `84.3313`，排名第 42 |

## 3. 已完成的本地流程

1. [x] 冻结原始数据、`v0_original`、`v1_scene_80_20`、模型权重和推理参数。
2. [x] 完成 direct 与 tiled 推理对照，确认切片入口可以处理固定验证集。
3. [x] 创建 Docker 交付目录：`Dockerfile`、Linux `environment.yml`、`app/`、`models/`。
4. [x] 让容器支持 `python /app/main.py --input /input --output /output`。
5. [x] 构建 `linux/amd64` 镜像 `weixing-submission:v1`。
6. [x] 使用 GPU 和 `--network none` 完成 897 张 Docker 全量运行。
7. [x] 检查 `/output/result.json` 的完整性、坐标格式和图片覆盖率。
8. [x] 用内部评估脚本核验 Recall、FDR 和单图耗时。
9. [x] 使用比赛页面提供的 ACR 登录命令完成镜像 push。
10. [x] 在比赛系统提交 `trial-v1.0` 并保存官方结果。

Docker 具体运行记录见 [`INF008_docker_full_validation.md`](INF008_docker_full_validation.md)。

## 4. 本地 Docker 全量结果

输入为固定验证集的 897 张图片，运行条件为 Docker Desktop Linux 引擎、GPU 和 `--network none`。

| 指标 | 结果 | 内部门槛 | 判断 |
| --- | ---: | ---: | --- |
| Overall Recall | 0.9655903842 | >= 0.85 | 通过 |
| Overall FDR | 0.0709750567 | <= 0.20 | 通过 |
| 最大单图耗时 | 1.800521 秒 | <= 20 秒 | 通过 |

总体 TP / FP / FN：`4097 / 313 / 146`。

内部评估脚本返回：

```json
{"recall_ge_0_85": true, "fdr_le_0_20": true, "latency_le_20s": true}
```

完整证据目录：

```text
submit/v1/test-output/docker_full_20260829/
```

其中的指标仍然是公开验证集内部结果，不是比赛封闭测试成绩。

## 5. 官方试运行结果

比赛系统返回的提交 ID 为 `2551`，tag 为 `trial-v1.0`，提交状态为 `ACCEPTED`，任务状态为“已完成”，且计入成绩。综合分为 `84.3313`，综合排名为第 42 名，提交时间为 `2026-08-29 19:01:44`，成绩时间为 `2026-08-29 19:12:45`。

页面可见的官方指标为：ship recall `0.899137`、ship false_detection_rate `0.129504`；aircraft recall `0.996229`、aircraft false_detection_rate `0.025811`；vehicle recall `0.939597`、vehicle false_detection_rate `0.274611`；average_inference_time_sec `1.837167`。完整记录见 [`SUB001_trial_2551.md`](../../submissions/official_feedback/SUB001_trial_2551.md)。

页面未展示整体 Recall 和整体虚警率的完整字段，报告不能从分组字段反推出整体值。官方平台的接受、完成和计分状态以平台记录为准。

## 6. 第一轮对照实验结论

### 6.1 训练基线

- `EXP001` 使用比赛方原始标签训练，实际记录 56 轮后由用户中断；
- 最佳轮次为第 55 轮，最佳模型已经保留；
- 训练结果可以支持后续推理、封装和提交链路验证；
- 本轮不因训练中断而补跑或覆盖原始输出。

### 6.2 后处理参数

在固定模型和验证协议下，已经完成 NMS IoU 与置信度阈值对照。总体趋势是：提高置信度或降低 NMS IoU 可以减少部分 FP，但会带来一定 FN；单纯调阈值不能解决发射车误检、舰船漏检和细分类混淆。

当前选择 `conf=0.30`、`NMS IoU=0.60`，原因是它在整体 Recall、整体 FDR 和后续 Docker 交付稳定性之间保持折中。该选择是第一轮提交候选，不是经过封闭测试确认的最终最优方案。

### 6.3 错误机制

全量错误分析显示，重点问题集中在：

- `FSC` 发射车误检率较高；
- `MS`、`QHS` 舰船类别存在漏检、误检和类别混淆；
- `A5_F-16`、`A13_F-15` 等飞机细分类存在相互混淆；
- 一部分错误可能来自原始标注质量，但本轮不使用人工审计结果干预对照实验。

这些问题保留给官方基线结果之后的 `AUDIT001`、`EXP002` 和 `submit/v2`。

## 7. 阶段性冻结范围

本版本采用“部分冻结”而不是最终冻结。

### 已冻结

- 原始数据只读原则；
- 原始标签版本 `v0_original`；
- 验证协议 `v1_scene_80_20`；
- 25 类类别顺序和结果 schema；
- `EXP001` 最佳权重及其 SHA-256；
- tiled 推理参数；
- `submit/v1` 的 `/input`、`/output` 接口和 Docker 入口；
- 本地 Docker 全量验证证据。

### 尚未冻结

- 后续是否启用人工审计标签；
- `submit/v2` 的模型、参数和改进路线。

## 8. 交付边界

应纳入 GitHub 的内容：

- `submit/v1/app/` 推理代码；
- `submit/v1/Dockerfile`；
- `submit/v1/environment.yml`；
- 版本 manifest、实验记录、数据协议和官方资料；
- 不含数据和权重的 README、命令和复现说明。

不纳入 GitHub 的内容：

- 训练数据、标签和验证图片；
- `submit/v1/models/best.pt`；
- `submit/v1/test-input/` 图片；
- `submit/v1/test-output/` 运行结果；
- `runs/` 下的训练和推理缓存；
- 代理配置、密码和 ACR 临时凭证。

## 9. 下一步

1. 保留 `SUB001` 官方结果作为第一轮原始标签对照基线。
2. 结合官方分组指标和已有错误分析，决定是否启用人工标注审计。
3. 如需改标、重新训练或改动推理参数，创建 `EXP002`/`submit/v2`，不覆盖 v1。

## 10. 报告可用结论

第一轮已经完成从原始数据基线、固定验证、推理参数对照、错误分析、提交入口、Docker 离线全量验证到官方试运行的完整链路。官方系统接受提交 `2551` 并完成计分，综合分为 `84.3313`，排名第 42。公开验证集内部结果和官方封闭测试结果已分开归档。
