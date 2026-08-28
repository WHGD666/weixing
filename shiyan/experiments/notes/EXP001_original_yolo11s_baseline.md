# EXP001：官方原始标签 YOLO11s baseline

状态：prepared，等待启动训练。

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

默认输出目录：

```text
runs/train/exp001_original_yolo11s_baseline/
```

重点关注：

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `val_batch*_pred.jpg`

这些训练输出属于可再生成或大文件结果，不直接纳入 Git。后续如果作为候选模型，需要复制摘要指标和模型哈希到实验登记表。

## 提交策略

第一版平台提交前必须先完成：

1. 本地训练完成。
2. 检查验证集指标和明显坏例。
3. 生成或确认推理入口能输出官方 `/output/result.json`。
4. Docker 中使用模型权重、类别表和配置，不依赖网络。
5. 本地 Docker `--network none` 冒烟测试通过。

如果仍在预测评阶段，可以用该模型优先测试 Docker 流程；若已经进入正式测评阶段，应谨慎使用 5 次提交机会。
