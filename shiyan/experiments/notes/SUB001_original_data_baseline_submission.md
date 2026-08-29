# SUB001：原始比赛数据基线提交候选

状态：**local_candidate_validated，等待比赛平台试运行**

## 1. 实验定位

在不进行人工修标、不引入外部数据、不改变冻结划分的前提下，把第一版模型封装为可运行 Docker 候选并提交，获得一个真实官方评测基准。当前本地部分已经完成，比赛系统正式结果尚未产生。

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
| 官方封闭测试 | 尚未提交，不能提前宣称成绩 |

## 3. 已完成的本地流程

1. [x] 冻结原始数据、`v0_original`、`v1_scene_80_20`、模型权重和推理参数。
2. [x] 完成 direct 与 tiled 推理对照，确认切片入口可以处理固定验证集。
3. [x] 创建 Docker 交付目录：`Dockerfile`、Linux `environment.yml`、`app/`、`models/`。
4. [x] 让容器支持 `python /app/main.py --input /input --output /output`。
5. [x] 构建 `linux/amd64` 镜像 `weixing-submission:v1`。
6. [x] 使用 GPU 和 `--network none` 完成 897 张 Docker 全量运行。
7. [x] 检查 `/output/result.json` 的完整性、坐标格式和图片覆盖率。
8. [x] 用内部评估脚本核验 Recall、FDR 和单图耗时。
9. [ ] 使用比赛页面提供的 ACR 登录命令完成镜像 push。
10. [ ] 在比赛系统提交 `trial-v1.0` 并保存官方结果。

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

## 5. 第一轮对照实验结论

### 5.1 训练基线

- `EXP001` 使用比赛方原始标签训练，实际记录 56 轮后由用户中断；
- 最佳轮次为第 55 轮，最佳模型已经保留；
- 训练结果可以支持后续推理、封装和提交链路验证；
- 本轮不因训练中断而补跑或覆盖原始输出。

### 5.2 后处理参数

在固定模型和验证协议下，已经完成 NMS IoU 与置信度阈值对照。总体趋势是：提高置信度或降低 NMS IoU 可以减少部分 FP，但会带来一定 FN；单纯调阈值不能解决发射车误检、舰船漏检和细分类混淆。

当前选择 `conf=0.30`、`NMS IoU=0.60`，原因是它在整体 Recall、整体 FDR 和后续 Docker 交付稳定性之间保持折中。该选择是第一轮提交候选，不是经过封闭测试确认的最终最优方案。

### 5.3 错误机制

全量错误分析显示，重点问题集中在：

- `FSC` 发射车误检率较高；
- `MS`、`QHS` 舰船类别存在漏检、误检和类别混淆；
- `A5_F-16`、`A13_F-15` 等飞机细分类存在相互混淆；
- 一部分错误可能来自原始标注质量，但本轮不使用人工审计结果干预对照实验。

这些问题保留给官方基线结果之后的 `AUDIT001`、`EXP002` 和 `submit/v2`。

## 6. 阶段性冻结范围

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

- ACR 远程镜像是否推送成功；
- 比赛封闭测试的官方 Recall、FDR、耗时和分数；
- 是否启用人工审计标签；
- `submit/v2` 的模型、参数和改进路线。

## 7. 交付边界

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

## 8. 下一步

1. 比赛系统开放后，执行页面生成的临时 ACR 登录命令。
2. 推送 `competition-registry.cn-beijing.cr.aliyuncs.com/competition/team614651:trial-v1.0`。
3. 在网页提交试运行，保存官方状态、成绩、日志和提交时间。
4. 官方结果归档后，再决定是否启用人工标注审计并创建 `EXP002`/`submit/v2`。

## 9. 报告可用结论

第一轮已经完成从原始数据基线、固定验证、推理参数对照、错误分析、提交入口到 Docker 离线全量验证的完整链路。本地公开验证集上三项内部门槛均通过，说明当前方案具备进入官方试运行的工程条件；但由于官方封闭测试尚未执行，不能据此宣称初赛通过或获得性能基础分。
