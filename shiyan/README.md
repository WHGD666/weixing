# shiyan 实验工程区

`shiyan/` 是本项目后续所有可复现实验的主工作区。这里先搭架构，不急着写模型代码。

当前阶段：**三次正式提交和 INF012 算法搜索已归档；EXP006 Data3 + YOLO11x 容量对照正在 RTX 5090 训练，暂不继续提交**。

截至 2026-09-04 的统一项目状态、版本映射和候选方案见 [`experiments/PROJECT_STATUS_20260904.md`](experiments/PROJECT_STATUS_20260904.md)。

## 核心原则

- 原始数据只读，不直接修改 `shiyan/data/`。
- 每个正式实验必须有唯一 `run_id`。
- 每个结果必须能追溯到 Git commit、配置、数据指纹、划分版本和运行命令。
- 训练/验证/holdout 划分冻结前，不比较模型优劣。
- 失败实验也要登记，不静默删除。
- 本地验证结果不能写成官方隐藏测试成绩。

## 目录总览

```text
shiyan/
├── project_contract/      # 任务、标签、指标、划分策略等协议
├── data_registry/         # 数据清单、指纹、审计报告、标签版本、划分版本
├── environment/           # Python 依赖清单、安装顺序、环境冻结记录
├── configs/               # 数据、协议、模型、训练、推理、评估配置
├── src/                   # 后续代码模块位置，目前只建目录
├── scripts/               # 可执行脚本入口位置，目前只建目录
├── tests/                 # 数据契约、指标、冒烟测试
├── experiments/           # 实验注册表、run manifest、对比记录、失败记录
├── artifacts/             # 结果证据区，默认只追踪目录和小型表格
├── submissions/           # 提交候选、冻结版本和官方反馈
├── reports/               # 研究报告、图表、答辩材料
├── docker/                # Docker 封装相关文件
├── notebooks/             # 探索和分析 notebook
└── data/                  # 本地数据，已被 Git 忽略
```

## 一次正式实验必须记录

最少记录：

- `run_id`
- 实验角色：diagnostic / scientific / competition / final
- Git branch 和 Git commit
- 数据版本和数据指纹
- 标签版本
- split 版本和 split 指纹
- 配置文件路径
- 完整运行命令
- 25 类细分类指标
- 三大类聚合指标
- 大图推理时间
- 失败状态或采纳/拒绝原因

## 当前阶段结论

- 原始 D0 与人工修订 D3 两套标签协议已完成逐图映射核验；D3 重训前仍需重建过期 manifest 和数据指纹。
- Model A/B/C 三份权重及哈希已冻结，897 张低阈值三模型缓存已完成。
- 正式 `v1.0/v2.0/v3.0` 均已归档；三次容器均正常执行，但硬性 Recall/FDR 尚未同时达标。
- 平台主要风险是 ship/vehicle FDR；aircraft 和推理时间相对稳定。
- 第一阶段算法搜索已完成，当前保留 P1 均衡候选、P2 保守候选和 P3 低 FDR 诊断线。
- Data3 已在独立 `overshiyan/` 工程中重新审计并冻结指纹，EXP006 使用官方 YOLO11x、1024、60 轮训练。
- 内部验证、试运行和正式平台结果严格分开记录。

## 下一步

1. 让 `EXP006-X0` 完成训练，不在运行中改配置或重复启动进程。
2. 保存并哈希 best、last 和有价值的周期权重，在固定 897 图上完成 D3/D0 双协议评估。
3. 用最差协议的三大类宏 Recall/FDR 选择候选，Ultralytics mAP 只作辅助。
4. 在 RTX 3090 24 GB 上完成显存、时间、schema 和 Docker 无网络验收后，才讨论下一次正式提交。
