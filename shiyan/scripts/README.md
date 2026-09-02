# scripts

## 本地推理与评估

当前脚本必须从仓库根目录、已激活的 `(weixing)` 环境中运行。下面的命令由项目成员在本机执行，Codex 不代替执行。

### 1. 抽样推理冒烟测试

```powershell
python shiyan/scripts/test_inference.py `
  --model runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/best.pt `
  --image-list shiyan/data_registry/split_assignments/v1_scene_80_20/val.txt `
  --output-dir runs/test/exp001_baseline_sample `
  --sample-count 12 `
  --mode direct `
  --save-vis
```

它会验证模型加载、类别编号、边界框范围、结果图片覆盖和 `result.json` 字段，并输出 `timings.json` 与可视化图片。

### 2. 抽样结果按比赛指标评估

```powershell
python shiyan/scripts/evaluate_official.py `
  --predictions runs/test/exp001_baseline_sample/result.json `
  --image-list runs/test/exp001_baseline_sample/image_list.txt `
  --timings runs/test/exp001_baseline_sample/timings.json `
  --output-dir runs/test/exp001_baseline_sample/metrics
```

这一步使用项目的 Metric Contract v0：FSC 使用 IoU 0.35，舰船和飞机使用 IoU 0.50，按置信度从高到低匹配，并统计 25 类及三大类的 TP、FP、FN、Recall、FDR。输出会同时记录历史兼容的三大类 pooled 平均、按细分类别宏平均的候选口径和对应 gate；抽样结果只能用于排查流程，不能用于判断比赛门槛。

可用 `--label-version`、`--split-version`、`--experiment-id`、`--run-id` 写入实验元数据，便于训练后验收时确认结果来自同一数据和同一实验。

### 2.1 新旧模型同条件对照

训练完成后，新旧模型必须使用同一验证集清单、同一标签版本、同一推理参数和同一评估脚本。分别评估后，用下面的脚本生成差值报告：

```powershell
python shiyan/scripts/compare_official_metrics.py `
  --baseline runs/test/exp002_old_model_data2/metrics/official_metrics.json `
  --candidate runs/test/exp002_new_model_data2/metrics/official_metrics.json `
  --output-dir runs/test/exp002_model_comparison
```

它会生成 `comparison.json`、`comparison.csv` 和 `comparison.md`，同时比较 Overall、三大类、两种宏平均口径以及 gate 候选。只有验证集数量、Metric Contract、IoU 规则和数据版本一致时，差值才有意义。

### 3. 全量错误分析

```powershell
python shiyan/scripts/analyze_errors.py `
  --predictions runs/test/exp001_baseline_val_direct_20260829/result.json `
  --image-list runs/test/exp001_baseline_val_direct_20260829/image_list.txt `
  --output-dir runs/test/exp001_baseline_val_direct_20260829/error_analysis `
  --top-images 30
```

它会生成误检/漏检图片排名、按类别错误统计，以及前 30 张问题图的可视化。绿色是已匹配预测框，橙色是误检框，红色是漏检真实框。先看 `error_per_class.csv` 和 `visualizations/`，再决定阈值或数据策略。

### 4. 类别相关阈值过滤

以 `conf=0.25、iou=0.60` 的原始预测为基础，只提高 FSC（类别 24）的阈值：

```powershell
python shiyan/scripts/filter_class_thresholds.py `
  --input runs/test/exp001_baseline_val_direct_nms060_20260829/result.json `
  --image-list runs/test/exp001_baseline_val_direct_nms060_20260829/image_list.txt `
  --timings runs/test/exp001_baseline_val_direct_nms060_20260829/timings.json `
  --output-dir runs/test/exp001_baseline_class_fsc035_nms060_20260829 `
  --class-threshold 24=0.35
```

该脚本只过滤指定类别，其他 24 类保持不变。它不重新运行模型，也不改变原始 `result.json`。

第二轮 `INF009` 使用该工具对固定 tiled 预测做 FSC（类别 24）阈值对照。实验记录、候选范围和选择门槛见 [`INF009_class_aware_vehicle_threshold.md`](../experiments/notes/INF009_class_aware_vehicle_threshold.md)。先生成候选，再对每个候选运行 `evaluate_official.py`；未通过选择门槛前不修改 `submit/v1`、不构建 `submit/v2`。

### 5. 导出标准 COCO 检测结果

```powershell
python shiyan/scripts/export_coco.py `
  --input runs/test/exp001_baseline_sample/result.json `
  --output runs/test/exp001_baseline_sample/coco_results.json
```

Docker 运行时使用 `result.json`，其中框是 `[x1,y1,x2,y2]`；作品材料中的 COCO 检测结果使用 `coco_results.json`，其中框转换为 `[x,y,width,height]`。两种文件均由代码生成，禁止手工修改坐标。

### 6. 整个固定验证集的 direct 推理

固定验证集不是一个物理目录，而是 `val.txt` 中的图片清单；当前统一入口直接读取清单，不需要复制数据。抽样测试通过后，使用相同脚本把 `sample-count` 改成验证集总数：

```powershell
python shiyan/scripts/test_inference.py `
  --model runs/detect/runs/train/exp001_original_yolo11s_baseline/weights/best.pt `
  --image-list shiyan/data_registry/split_assignments/v1_scene_80_20/val.txt `
  --output-dir runs/test/exp001_baseline_val_direct `
  --sample-count 897 `
  --mode direct
```

全量推理完成后，再将 `result.json` 和 `timings.json` 交给 `evaluate_official.py`，并使用同一输出目录自动保存的 `image_list.txt`。不要把训练集目录直接当作官方输入目录。

### 7. 大图切片测试

在抽样 direct 测试通过后，再执行相同入口的 `--mode tiled`，并单独记录 `tile-size`、`tile-overlap`、`merge-iou`、最大单图耗时和框数量变化。切片参数改变就属于新的测试配置，不能和 direct 结果混为一个实验。

可执行命令入口位置。正式实验记录中应写明从哪个脚本启动。

### 8. v2 图像模态辅助诊断

该实验固定使用当前最高分 v2 模型 `submit/v2/models/best.pt` 和 v2 的 tiled `Detector`，只在模型输出之后按整张图的黑白/彩色特征调整类别阈值，不修改标签，也不修改 `submit/v2`。默认策略是：黑白图降低船只阈值，彩色图提高船只阈值；飞机和 FSC 保持 v2 的阈值。

先用 12 张固定图片检查流程：

```powershell
python shiyan/scripts/run_v2_modality_experiment.py `
  --model submit/v2/models/best.pt `
  --image-list shiyan/data_registry/split_assignments/v1_scene_80_20/val.txt `
  --output-dir runs/test/EXP003_v2_modality_smoke `
  --sample-count 12
```

`--sample-count 12` 会从固定清单中均匀抽取 12 张图；省略该参数或设置为 `0` 就运行清单中的全部图片。全量运行命令：

```powershell
python shiyan/scripts/run_v2_modality_experiment.py `
  --model submit/v2/models/best.pt `
  --image-list shiyan/data_registry/split_assignments/v1_scene_80_20/val.txt `
  --output-dir runs/test/EXP003_v2_modality_soft `
  --strategy soft `
  --conf 0.10 `
  --ship-gray-conf 0.20 `
  --ship-color-conf 0.60 `
  --aircraft-conf 0.30 `
  --fsc-conf 0.35
```

脚本会生成 `result.json`、`timings.json`、`image_list.txt`、`modality_by_image.csv` 和 `modality_summary.json`。评估方式与普通候选相同：将这些结果交给 `evaluate_official.py`。`--strategy strict` 会把彩色图中的船只预测全部删除，只作为压力对照，不作为默认提交方案。若需要严格复现 v2 基线，应直接评估已有 v2 结果，不把这个诊断脚本的结果冒充 v2 基线。

目录：

- `audit/`：数据审计。
- `train/`：训练入口。
- `infer/`：推理入口。
- `eval/`：评估入口。
- `packaging/`：提交包生成和冒烟测试。
