# RTX 3090 submission skeleton

This image uses the same inference source as local validation and a conservative
`tile_batch=1` default for 24 GB VRAM. EXP006 `best.pt` has been frozen into
`submit/models/best.pt` and is the next candidate for Docker validation and one
formal platform upload.

Frozen candidate:

- name: `EXP006_X0_best`;
- model SHA256:
  `111ae87e2911dcf35a68cb0b7d047e75f88ab62baa7367e690324133bbf2d67d`;
- internal worst D0/D3 Recall/FDR: `0.885259 / 0.185691`;
- RTX 5090 max-image validation time: `4.284 s`;
- status: Python smoke, Docker build, Docker no-network smoke, and Docker
  no-network 897-image validation passed locally on RTX 4060; RTX 3090 timing is
  still the deployment reference if available.

Build from the `overshiyan` root:

```powershell
docker build -f submit/Dockerfile -t weixing-submission:x6 .
```

Before a formal push, run smoke and full validation with `--network none`, verify
`result.json`, record the image ID/digest, and benchmark the largest images on an
actual RTX 3090. Training-host timing is not deployment evidence.

If this candidate is uploaded to the competition platform, use the next platform
tag shown by the web page, expected to be `v4.0` after formal submissions
`v1.0-v3.0`.
