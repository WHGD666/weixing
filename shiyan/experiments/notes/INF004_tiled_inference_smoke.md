# INF004：切片推理冒烟测试

状态：completed

## 实验目的

验证官方大图场景所需的 tiled 推理路径能够正常运行、还原坐标并生成完整的 `result.json`。本实验是工程接口测试，不是新的模型调参实验。

## 配置

| 项目 | 值 |
| --- | --- |
| 模型 | `EXP001` 的 `best.pt` |
| 数据 | `v1_scene_80_20` 验证清单抽样 12 张 |
| 推理模式 | tiled |
| tile-size | 1024 |
| tile-overlap | 0.20 |
| merge-iou | 0.50 |
| tile-batch | 4 |
| conf | 0.25，脚本默认值 |
| NMS IoU | 0.70，脚本默认值 |
| 数据指纹 | `5ba10a4296c997097abcb51f8d4633205043cc8e216d2191468a0ceb16924962` |
| split 指纹 | `126b297c2706c71b8be8df4e3736b9b30871f18673147df1c3e31e3997f8ec77` |

## 用户执行结果

- validated_images：12
- validated_objects：52
- total_seconds：3.988 秒
- max_image_seconds：3.567 秒
- result：`runs/test/SUB001_tiled_smoke_20260829/result.json`
- image_list：`runs/test/SUB001_tiled_smoke_20260829/image_list.txt`
- 结果文件和覆盖率校验：通过

## 内部指标核验

- Overall TP / FP / FN：46 / 6 / 0
- Overall Recall：1.000000
- Overall FDR：0.115385
- ship Recall/FDR：1.000000 / 0.000000
- aircraft Recall/FDR：1.000000 / 0.119048
- vehicle Recall/FDR：1.000000 / 0.166667
- latency gate：通过

与 direct 冒烟测试相比，切片模式少 1 个预测框、FDR 更低且耗时更低。由于样本只有 12 张，该结果只证明工程链路可用；下一步使用提交候选的 `conf=0.30、NMS IoU=0.60` 做一次 897 张全量切片验证。

## 结论

切片推理路径已具备继续验证条件，未发现启动、输出格式或结果覆盖率问题。当前不构建 Docker，不推送 Git，不把本次抽样指标写成官方成绩。
