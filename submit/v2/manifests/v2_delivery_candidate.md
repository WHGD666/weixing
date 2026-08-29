# v2 交付候选清单

状态：**docker_full_validated，等待推送和官方试运行**

## 版本边界

- 交付目录：`submit/v2`；
- 基线：`submit/v1`；
- 选择实验：`INF009-fsc35`；
- 模型权重：与 v1 相同，SHA-256 为 `6A26043F7961B037808CC8AC383B53203B7914716726346087F76603406E4F54`；
- 标签：不参与运行时，保持 `v0_original` 研发基线；
- 代码变化：仅增加类别 24 `FSC` 的阈值 `0.35`。

## 运行参数

```text
mode=tiled
imgsz=1024
conf=0.30
iou=0.60
tile-size=1024
tile-overlap=0.20
merge-iou=0.50
tile-batch=4
max-det=300
fsc-conf=0.35
```

## Docker 验证要求

- 构建平台固定为 `linux/amd64`；
- 运行时必须使用 GPU；
- 全量复核使用 `--network none`；
- 输入挂载到 `/input`，输出挂载到 `/output`；
- 必须生成 `/output/result.json` 和 `/output/timings.json`；
- 复核结果必须通过 schema、三大类平均 gate 和延迟 gate；
- v2 验证期间不覆盖 v1 镜像、目录或官方 tag。

## Windows 全量入口证据

- 输入：897 张固定验证图片；
- 输出：`submit/v2/test-output/app_full_20260830/`；
- TP / FP / FN：`4097 / 311 / 146`；
- Overall Recall / FDR：`0.965590 / 0.070554`；
- Group-mean Recall / FDR：`0.882586 / 0.191804`；
- 总耗时：`34.364042s`；
- 最大单图耗时：`4.010357s`；
- schema、三大类平均 gate 和延迟 gate 均通过；
- 结果与 INF009-fsc35 离线过滤结果一致。

## Docker 全量复核

- 镜像：`weixing-submission:v2`；
- 平台：`linux/amd64`；
- 运行：GPU，`--network none`；
- 输入：897 张固定验证图片；
- 输出：`submit/v2/test-output/docker_full_20260830/`；
- TP / FP / FN：`4097 / 311 / 146`；
- Overall Recall / FDR：`0.965590 / 0.070554`；
- Group-mean Recall / FDR：`0.882586 / 0.191804`；
- 总耗时：`76.074987s`；
- 最大单图耗时：`1.903421s`；
- schema、三大类平均 gate 和延迟 gate 均通过；
- 结果与 Windows 入口及 INF009-fsc35 离线结果一致；
- 当前仅完成本地复核，尚未推送 ACR，尚未官方提交。
