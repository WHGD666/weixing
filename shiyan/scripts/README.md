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

这一步使用项目的 Metric Contract v0：FSC 使用 IoU 0.35，舰船和飞机使用 IoU 0.50，按置信度从高到低匹配，并统计 25 类及三大类的 TP、FP、FN、Recall、FDR。抽样结果只能用于排查流程，不能用于判断比赛门槛。

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

目录：

- `audit/`：数据审计。
- `train/`：训练入口。
- `infer/`：推理入口。
- `eval/`：评估入口。
- `packaging/`：提交包生成和冒烟测试。
