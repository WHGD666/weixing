# shiyan 实验工程区

`shiyan/` 是本项目后续所有可复现实验的主工作区。这里先搭架构，不急着写模型代码。

当前阶段：**v2 官方试运行已完成，当前进行 AUDIT002 的 FSC/MS 定向标签审计**。

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

- 原始数据、原始标签、25 类映射和 `v1_scene_80_20` 划分已经建立并冻结。
- `EXP001` 原始标签基线训练已完成 56 轮记录，最佳模型已保留。
- direct、tiled、提交入口和 Docker `--network none` 全量链路已经验证。
- 本地内部指标与官方封闭测试结果分开记录；`SUB001` 官方提交 `2551` 已完成，综合分 `84.3313`，排名第 42。

## 下一步

1. 保留 `SUB001`、`SUB002` 官方结果作为原始标签和 v2 对照基线。
2. 按 `AUDIT002` 队列完成 FSC/MS 前 30 张问题图人工复核。
3. 只有确认存在集中标签问题，才复制出 `v1_audited` 并创建 `EXP002`；不覆盖 `v0_original`、`submit/v1` 或 `submit/v2`。
