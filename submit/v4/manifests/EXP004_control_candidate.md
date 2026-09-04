# EXP004 control submission candidate

状态：`formal_result_received_frozen`

## 身份

- 候选 ID：`EXP004_original_continue40_control`
- 训练实验：`EXP004_original_continue40`
- 训练标签：`v0_original`
- 固定划分：`v1_scene_80_20`
- 模型：`submit/v4/models/best.pt`
- 模型 SHA-256：`90B39F94D272648781152F28EF840B553D1AF64441815522319E7461C756048B`
- 类别数：25
- 类别顺序：沿用 `submit/v2/app/labels.py`

## 推理配置

```yaml
mode: tiled
imgsz: 1024
conf: 0.10
iou: 0.60
max_det: 300
tile_size: 1024
tile_overlap: 0.20
merge_iou: 0.50
tile_batch: 4
ship_conf: 0.30
aircraft_conf: 0.30
fsc_conf: 0.35
```

## 内部依据

EXP004 control 在固定 897 张验证集上的历史分组口径结果：

- group pooled mean Recall：`0.8653839719`
- group pooled mean FDR：`0.1484837948`
- 最大单图耗时：`3.530787 s`
- TP / FP / FN：`4079 / 293 / 164`

这些指标来自公开固定验证集，仅用于 Docker 候选筛选，不代表官方隐藏测试结果。

## 验收与正式结果

1. Docker 构建成功，入口在无网络容器中完成 12 张 smoke。
2. 固定 897 张验证集全量输出、类别编号与 schema 检查通过。
3. Docker 全量内部宏 Recall/FDR 为 `0.8653839719/0.1484837948`，最大单图 `2.009783 s`。
4. 镜像以平台 tag `v3.0` 推送，digest 为 `sha256:bed644d28ecc3e436d2c4dc582e8f4594cc520f2957735a3e533f35e18043e35`。
5. 正式提交 ID `3872`：宏 Recall `0.846539`、宏 FDR `0.262158333`、时间 `1.742 s`，未同时通过精度硬门槛。

该候选已经冻结，不再用同一配置重复提交。正式复盘见 `shiyan/submissions/official_feedback/FORMAL003_v3_3872.md`。
