# RTX 3090 submission skeleton

This image uses the same inference source as local validation and a conservative
`tile_batch=1` default for 24 GB VRAM. It is intentionally not buildable until an
evaluated model is frozen into `submit/models/best.pt`.

Build from the `overshiyan` root:

```powershell
docker build -f submit/Dockerfile -t weixing-submission:x6 .
```

Before a formal push, run smoke and full validation with `--network none`, verify
`result.json`, record the image ID/digest, and benchmark the largest images on an
actual RTX 3090. Training-host timing is not deployment evidence.
