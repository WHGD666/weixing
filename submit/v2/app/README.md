# submit/v2/app

这里放第二轮 Docker 镜像随附的候选推理代码。它基于 `submit/v1`，只增加 INF009 选定的 FSC 类别阈值后处理。

必须包含入口 `main.py`，并支持：

```bash
python /app/main.py --input /input --output /output
```

默认运行配置：

- `mode=tiled`
- `tile-size=1024`
- `tile-overlap=0.20`
- `merge-iou=0.50`
- `conf=0.30`
- `iou=0.60`
- `fsc-conf=0.35`，仅作用于类别 24 `FSC`

入口启动时会检查 CUDA GPU，模型默认从 `/app/models/best.pt` 加载，并将结果写入 `/output/result.json`。FSC 过滤发生在 tiled 跨窗口合并完成之后，与 INF009 离线过滤保持一致。
