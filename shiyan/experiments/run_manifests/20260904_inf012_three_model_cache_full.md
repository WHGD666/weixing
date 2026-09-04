# Run Manifest：20260904 INF012 三模型全量缓存

| 字段 | 值 |
| --- | --- |
| run_id | `20260904_inf012_three_model_cache_full` |
| status | `completed` |
| task | `detect/cache` |
| experiment_role | `competition_diagnostic` |
| protocol_version | `D0 v1_scene_80_20 + D3 mapped evaluation` |
| working_directory | `D:\daima\weixing` |
| git_branch / commit | `main` / `4daedca9a65aaecb3c1481279b5be52da2d8c6a9` |
| git_dirty | `true` |
| holdout_evaluated | `false` |
| Python | `3.10.20` |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB` |
| driver | `581.57` |

## 命令

```powershell
python shiyan/scripts/cache_multi_model_predictions.py `
  --model "a=submit/v2/models/best.pt" `
  --model "b=runs/detect/runs/train/exp004_original_continue40/weights/best.pt" `
  --model "c=runs/detect/runs/train/exp005_data3_manual_revision_classfix-2/weights/best.pt" `
  --image-list shiyan/data_registry/split_assignments/v1_scene_80_20/val.txt `
  --output-dir runs/test/INF012_three_model_cache_full_20260904
```

未显式覆盖的推理参数使用脚本冻结默认值：`imgsz=1024`、`conf=0.05`、`iou=0.60`、`mode=tiled`、`tile_size=1024`、`tile_overlap=0.20`、`merge_iou=0.50`、`tile_batch=4`。

## 输入哈希

| 输入 | SHA256 |
| --- | --- |
| Model A | `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54` |
| Model B | `90B39F94D272648781152F28EF840B553D1AF64441815522319E7461C756048B` |
| Model C | `3D3C8CB2B256B127668EC113F50DA7C261A295C30C7AEB2A2E177D2682D5A0E5` |
| image_list.txt | `15A48AEA28D46813B502ABB3FD9748058A7094BEC658425611A14454B5D37DEC` |

## 输出与时间

- 图像：897；模型：A/B/C；模态：灰度 266、彩色 627、不确定 4。
- 总执行时间：`82.557 s`。
- 三模型累计最大单图代理时间：`5.257 s`。
- `raw_cache.json`：4,557,143 字节，SHA256 `D1C1E3E21B0C9E1ACF5D1BF43C01699162383533F4513E84E685711692C43A4A`。
- 本地路径：`runs/test/INF012_three_model_cache_full_20260904/`。

`runs/` 不进入 Git；本 manifest、模型哈希、缓存哈希、命令和候选对比表进入 Git。该缓存只用于内部固定验证集算法搜索，不是官方隐藏测试结果。
