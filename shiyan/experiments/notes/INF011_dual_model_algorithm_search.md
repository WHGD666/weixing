# INF011：双模型算法后处理搜索

状态：**smoke_passed，双模型方案已由 INF012 扩展为 A/B/C 三模型双标签协议**

本实验不修改标签、不重新训练模型、不覆盖既有 Docker 镜像，也不立即提交平台。目标是在固定 `v1_scene_80_20` 验证集上建立可复用的低阈值双模型预测缓存，再用纯 CPU 后处理快速验证组内去重、图像模态路由、按大类选模型和双模型一致性过滤。所有本地结论都必须与正式平台结果分开记录。

## 1. 正式平台事实

| 正式提交 | 实际模型/方案 | ship R/FDR | aircraft R/FDR | vehicle R/FDR | 宏平均 R/FDR | 时间 |
| --- | --- | --- | --- | --- | --- | ---: |
| `v1.0` / `3496` | 原始数据第一版模型（项目内称 v2） | 0.699474 / 0.251032 | 0.959178 / 0.061442 | 0.873684 / 0.371212 | 0.844112 / 0.227895 | 1.7846 s |
| `v2.0` / `3720` | 人工修订 data2 从头训练（项目内称 v3） | 0.284132 / 0.174486 | 0.901458 / 0.182432 | 0.705263 / 0.524823 | 0.630284 / 0.293914 | 1.7638 s |
| `v3.0` / `3872` | EXP004 续训模型 + control | 0.697666 / 0.364222 | 0.957740 / 0.058617 | 0.884211 / 0.363636 | 0.846539 / 0.262158 | 1.7420 s |

当前正式最好 Recall 距离 `0.85` 仅差 `0.003461`，但 FDR 距离 `0.20` 仍差 `0.062158`。因此本轮主目标是降低 ship/vehicle FDR，同时保护三大类宏平均 Recall；综合分只作参考。

## 2. 冻结输入

| 标识 | 权重 | 角色 |
| --- | --- | --- |
| Model A | `submit/v2/models/best.pt` | 正式 `v1.0` 对应的原始模型 |
| Model B | `runs/detect/runs/train/exp004_original_continue40/weights/best.pt` | 正式 `v3.0` 对应的 EXP004 续训模型 |
| 验证划分 | `shiyan/data_registry/split_assignments/v1_scene_80_20/val.txt` | 固定 897 张，不重新抽样 |
| 标签 | `v0_original` | 只读 |
| 基础推理 | tiled，1024，overlap 0.20，NMS IoU 0.60，merge IoU 0.50 | 所有候选相同 |
| 缓存阈值 | global conf 0.05 | 为离线阈值搜索保留低分框 |

Model A SHA-256：`6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54`。Model B SHA-256：`90B39F94D272648781152F28EF840B553D1AF64441815522319E7461C756048B`。缓存脚本会重新保存两个模型的实际 SHA-256，后续以缓存元数据为准。

## 3. 实现文件

- `shiyan/scripts/cache_dual_model_predictions.py`：每张图分别运行 Model A/B，一次性保存低阈值原始预测、模态统计、模型哈希和耗时。
- `shiyan/scripts/run_algorithm_postprocess.py`：不再调用 GPU，读取缓存并生成不同算法候选的标准 `result.json`。
- `shiyan/src/inference/algorithmic_postprocess.py`：双模型匹配、按大类路由、支持度阈值、双向模态过滤和组内 NMS 的纯 Python 核心。
- `shiyan/tests/test_algorithmic_postprocess.py`：保护跨细类组内去重、路由、融合、模态过滤和一致性阈值语义。

缓存中的图像模态分为 `grayscale`、`color`、`uncertain` 三种。`uncertain` 不执行硬删除，避免压缩噪声或弱彩色图被错误路由。

## 4. 算法因素

1. **组内 NMS**：原检测器只压制同一细类别框；本实验在 ship、aircraft、vehicle 三个大类内部跨细类别去重，目标是清理同一目标被预测成两个船型或两个机型造成的 FP。
2. **双向模态约束**：不仅提高彩色图 ship 阈值，也提高灰度图 aircraft/vehicle 阈值；`strict` 才进行硬删除，`soft` 只提高阈值。
3. **大类模型路由**：根据正式结果，默认 ship/aircraft 取 A，vehicle 取 B。该策略保留 A 的船只优势和 B 的车辆优势。
4. **双模型一致性**：同大类且 IoU 达标的 A/B 框融合并记为 support=2；一致框允许低阈值，单模型独有框使用高阈值，以降低 FDR。
5. **intersection 压力测试**：只保留双模型一致框，用来测量可达到的 FDR 下界，不直接预设为提交候选。

## 5. 实验顺序

必须先完成两个基线复现，随后一次只增加一个因素：

| 顺序 | 候选 | 唯一变化 | 用途 |
| ---: | --- | --- | --- |
| 0A | `source-a-control` | 无 | 复现原始模型本地基线 |
| 0B | `source-b-control` | 模型 A 改为 B | 复现 EXP004 本地基线 |
| 1 | `source-a-group-nms` | 只开启组内 NMS | 判断跨细类重复框是否是 FDR 来源 |
| 2 | `source-a-strict-modality` | 只开启双向 strict 模态约束 | 验证颜色域先验的上限与风险 |
| 3 | `route-aab` | ship=A、aircraft=A、vehicle=B | 验证按大类选模型 |
| 4 | `consensus-soft` | 一致框低阈值、独有框高阈值 | 主精度候选 |
| 5 | `intersection-strict` | 只保留一致框 | FDR 下界与 Recall 损失诊断 |

只有单因素结果明确后，才组合表现有效的因素；不得直接依据 12 张 smoke 的高分做选择。

## 6. 选择标准

固定验证集候选进入 Docker 前至少满足：

- 三大类宏平均 Recall `>= 0.85`；
- 三大类宏平均 FDR `<= 0.20`；
- 最大单图代理耗时 `< 20 s`；
- 相对复现基线的改善能够解释，且任一大类 Recall 不出现不可接受的坍塌；
- 优先目标为本地宏平均 Recall `>= 0.90`、FDR `<= 0.12`，给已观测到的平台分布偏移留安全余量；
- 同一候选必须保留 `configuration.json`、`result.json`、`timings.json` 和 `official_metrics.json`。

正式平台只剩两次提交机会。本轮本地搜索、Docker 验收和误差分析完成之前，不生成正式 tag、不 push、不点击提交。

## 7. 当前验证状态

- 新增核心模块通过 `py_compile`。
- 新增 6 个纯 Python 单元测试全部通过。
- 12 张真实模型双缓存 smoke 已通过：灰度 4 张、彩色 7 张、uncertain 1 张；双模型最大单图耗时 `5.028 s`。
- smoke 的 Model A control 为 TP/FP/FN=`46/4/0`，Model B control 为 `46/2/0`，两者 schema、图片顺序与指标评估均通过。
- smoke 中 A 的最大单图 `4.915 s` 主要包含首次 CUDA 预热；B 在 A 之后运行，最大单图 `0.113 s`。该顺序差异不能用于比较模型速度。
- 12 张样本对三个大类的支持量过小，不用于算法选择或泛化结论。
- INF011 未单独执行 897 张双模型全量缓存；其能力已由 INF012 的 897 张 A/B/C 缓存覆盖。
- INF012 已完成 D0/D3 双协议基线、路由、共识、投票及阈值扫描，当前均衡候选最差 Recall/FDR 为 `0.883983/0.165283`。
- 本记录冻结为双模型方案的前置设计与 smoke 证据，后续结论统一见 `INF012_three_model_dual_protocol_search.md`。
- 尚未为 INF012 候选构建新的提交镜像。
