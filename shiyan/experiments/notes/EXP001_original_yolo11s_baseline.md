# EXP001：官方原始标签 YOLO11s baseline

状态：interrupted，已完成 56 轮并由用户手动中断；最佳模型已保存。

本次运行的不可变记录见：
`shiyan/experiments/run_manifests/20260828_1940_detect_exp001_original_yolo11s_9e250460.md`

## 实验目标

用官方原始标签训练第一版可用检测模型，快速获得一条完整 baseline。该实验的目的不是追求最高分，而是打通以下链路：

- 原始数据审计结果可追溯。
- train/val 划分固定。
- YOLO 训练能够跑通。
- 验证集结果可以保存和复查。
- 后续能够接入 Docker 推理入口并参加预测评或正式测评。

## 数据与协议

| 项目 | 版本 |
| --- | --- |
| 标签版本 | `v0_original` |
| 数据审计版本 | `v0_original` |
| split 版本 | `v1_scene_80_20` |
| split 指纹 | `126b297c2706c71b8be8df4e3736b9b30871f18673147df1c3e31e3997f8ec77` |
| 数据配置 | `shiyan/configs/dataset/weixing_v1_scene_80_20.yaml` |
| 训练配置 | `shiyan/configs/train/exp001_original_yolo11s_baseline.yaml` |

## 训练策略

- 模型：`yolo11s.pt`
- 输入尺寸：`1024`
- epoch：`80`
- batch：`4`
- AMP：开启
- optimizer：Ultralytics `auto`
- seed：`20260828`

若显存不足，第一优先级是把 batch 降为 `2`；不要改变 split、标签版本或数据目录。

## 运行命令

在已激活的 `(weixing)` 环境和仓库根目录执行：

```powershell
yolo detect train cfg=shiyan/configs/train/exp001_original_yolo11s_baseline.yaml
```

若出现 OOM：

```powershell
yolo detect train cfg=shiyan/configs/train/exp001_original_yolo11s_baseline.yaml batch=2
```

## 预期输出

本次实际输出目录：

```text
runs/detect/runs/train/exp001_original_yolo11s_baseline/
```

重点关注：

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `val_batch*_pred.jpg`

这些训练输出属于可再生成或大文件结果，不直接纳入 Git。后续如果作为候选模型，需要复制摘要指标和模型哈希到实验登记表。

## 实际运行结果

| 项目 | 结果 |
| --- | --- |
| 运行状态 | 用户手动中断 |
| 实际完成轮数 | 56 |
| 当前最佳轮次 | 55 |
| 最佳 Precision | 0.94982 |
| 最佳 Recall（Ultralytics 验证指标） | 0.95606 |
| 最佳 mAP50 | 0.96406 |
| 最佳 mAP50-95 | 0.77669 |
| 最近一轮 mAP50-95 | 0.77206 |
| 训练耗时（results.csv 记录） | 8459.58 秒，约 2 小时 21 分 |
| 最佳权重 | `runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/best.pt` |
| 最近权重 | `runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/last.pt` |

第 55 轮为当前开发集上的最佳轮次，因此后续推理、可视化和 Docker 冒烟测试优先使用 `best.pt`。第 56 轮的 `last.pt` 仅用于必要时恢复训练。

这里的 Recall、mAP 等是 Ultralytics 内部验证指标，不等同于比赛封闭测试集上的官方召回率、虚警率或推理时间。当前实验也尚未生成官方测试结果，因此不能据此判断是否达到比赛刚性门槛。

## 当前结论

- 训练链路已经跑通，固定的场景级 8:2 划分能够正常使用。
- 原始标签基线已经获得可用于后续推理验证的模型，暂不对原始标注质量作最终判断。
- 该模型不作为最终提交模型冻结，只作为后续推理接口、评估器和标注审计实验的起点。
- 原始数据、标签版本、划分文件和本地训练输出保持分离，后续实验不得覆盖本次结果。

## 下一步准备

1. 用 `best.pt` 对少量训练集和验证集图像做推理，检查类别、框位置、漏检和重复框。
2. 建立与比赛规则一致的本地评估流程，单独统计类别匹配、IoU 阈值、召回率和虚警率。
3. 记录大图推理耗时，并确认后续切片推理方案是否满足 10000×10000 像素图像的时间约束。
4. 完成上述检查后，再决定是否进入 Docker 推理入口准备；当前不打包、不上传模型。

## 提交策略

第一版平台提交前必须先完成：

1. 本地训练完成。
2. 检查验证集指标和明显坏例。
3. 生成或确认推理入口能输出官方 `/output/result.json`。
4. Docker 中使用模型权重、类别表和配置，不依赖网络。
5. 本地 Docker `--network none` 冒烟测试通过。

如果仍在预测评阶段，可以用该模型优先测试 Docker 流程；若已经进入正式测评阶段，应谨慎使用 5 次提交机会。

## 报告图片素材

- [EXP001 训练曲线](../evidence/EXP001_original_baseline/training_curves.png)
- [EXP001 训练样本与标签示例](../evidence/EXP001_original_baseline/train_batch0.jpg)
