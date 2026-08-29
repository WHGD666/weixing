# submit/v1/app

这里放第一轮 Docker 镜像随附的最终推理代码。代码已经从本地验证过的 tiled 推理逻辑整理为自包含入口，并已随 `submit/v1` 镜像完成 Docker 全量验证。

必须包含入口 `main.py`，并支持：

```bash
python /app/main.py --input /input --output /output
```

入口需要实现或保留等价的 GPU 启动检查、模型单次加载、第一层图片读取、tiled 推理、结果写出和异常处理。代码中禁止出现 Windows 盘符路径，模型路径使用 `/app/models/`。

当前入口代码、模型构建上下文和环境文件均已准备完成。模型权重只保留在本地 Docker 构建上下文，不进入 GitHub。

默认运行配置：

- `mode=tiled`
- `tile-size=1024`
- `tile-overlap=0.20`
- `merge-iou=0.50`
- `conf=0.30`
- `iou=0.60`

入口启动时会检查 CUDA GPU，模型默认从 `/app/models/best.pt` 加载，并将结果写入 `/output/result.json`。
