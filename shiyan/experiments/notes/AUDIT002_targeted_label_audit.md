# AUDIT002：FSC/MS 定向标签审计

状态：**in_progress，已生成只读人工复核队列，尚未修改任何标签**

## 1. 目标

基于已经完成官方试运行的 v2 预测，优先检查 `FSC` 和 `MS` 相关的误检、漏检、错框和错类，判断哪些问题来自当前标签，哪些只是模型能力不足。只有经过原图、原始标签和预测框三方对照，才能决定是否建立新的审计标签版本。

## 2. 固定条件

| 项目 | 固定值 |
| --- | --- |
| 对照版本 | `submit/v2` / 官方 `trial-v2.0` / 提交 2750 |
| 预测文件 | `submit/v2/test-output/docker_full_20260830/result.json` |
| 固定划分 | `v1_scene_80_20`，897 张 |
| 标签版本 | `v0_original`，只读，不覆盖 |
| 审计目标 | FSC（类别 24）、MS（类别 3） |
| 当前目的 | 人工确认标签问题，不据此直接调参或提交 |

注意：897 张图片由冻结清单指定，实际文件位于 `shiyan/data/images/train`；对应标签位于 `shiyan/data/labels/train`。文件夹名不改变冻结划分含义。

## 3. 审计输出

运行队列生成器后，输出目录为 `runs/test/AUDIT002_v2_targeted_20260830/`：

- `review_queue.csv`：按 FSC/MS 错误优先级排序的逐图复核表，包含 `pending`、`decision`、`notes` 空栏；
- `visualizations/`：前 30 张问题图，只绘制 FSC/MS 的 TP、FP 和 FN；
- `audit_summary.json`：输入范围、目标类别和错误统计。

队列是辅助工具，不是标签修改结果。橙色框表示相对当前标签的 FP，红色框表示当前标签下的 FN，绿色框表示匹配成功；任何颜色都不能单独证明标签错误。

## 4. 人工判定规则

逐图打开原图、对应原始 `.txt` 标签和可视化，`review_queue.csv` 每行只填一个结论：

- `confirmed_label_issue`：确认漏标、错类、错框或重复标注；
- `model_error`：原始标签合理，预测错误；
- `ambiguous`：图像分辨率、目标边界或类别定义不足以判断；
- `reviewed_no_change`：已核对，当前标签无需修改。

先审排名前 30 张，不扩展到全量。不能根据预测框自动反向生成标签，也不能为了提高本地分数而修改模糊样本。

## 5. 后续标签版本规则

只有当审计结果显示存在集中、明确、可复现的标签问题时，才复制出新的 `v1_audited` 标签版本。原始 `v0_original` 永久保留，审计修改必须有：

1. 修改清单，记录文件、旧标签、新标签、原因和审计状态；
2. 新旧标签统计和数据指纹；
3. 与 `v0_original` 相同的冻结场景划分；
4. 只用一轮 `EXP002` 对照训练验证，不能把审计集直接当作最终成绩保证。

若 30 张中主要是模型错误或无法判断，则不生成 `v1_audited`，保留 v2，并继续分析模型结构或增加针对性训练数据。

## 6. 当前命令

```powershell
conda activate weixing
cd D:\daima\weixing

$auditOut = "runs/test/AUDIT002_v2_targeted_20260830"
python .\shiyan\scripts\build_label_audit_queue.py `
  --predictions "submit/v2/test-output/docker_full_20260830/result.json" `
  --image-list "submit/v1/test-output/app_full_20260829/image_list_original.txt" `
  --output-dir $auditOut `
  --target-class 24 `
  --target-class 3 `
  --top-images 30
```

本命令只读图片、标签和预测，不改原始数据，不调用 GPU，不构建 Docker。
