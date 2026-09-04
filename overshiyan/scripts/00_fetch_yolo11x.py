from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import sha256_file, utc_now, write_json

EXPECTED_SHA256 = "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"


def main() -> None:
    from ultralytics import YOLO

    destination = ROOT / "models/pretrained/yolo11x.pt"
    if destination.is_file():
        candidate = destination
    else:
        os.chdir(ROOT)
        model = YOLO("yolo11x.pt")
        candidates = [
            Path(str(getattr(model, "ckpt_path", ""))),
            ROOT / "yolo11x.pt",
        ]
        candidate = next((path.resolve() for path in candidates if path and path.is_file()), None)
        if candidate is None:
            raise FileNotFoundError("Ultralytics completed without exposing the downloaded yolo11x.pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if candidate.parent == ROOT and candidate.name == "yolo11x.pt":
            shutil.move(candidate, destination)
        else:
            shutil.copy2(candidate, destination)
        candidate = destination

    model = YOLO(str(candidate))
    scale = model.model.yaml.get("scale")
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    if scale != "x" or parameter_count < 50_000_000:
        raise ValueError(f"downloaded checkpoint is not YOLO11x: scale={scale}, parameters={parameter_count}")
    digest = sha256_file(candidate)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"unexpected yolo11x.pt SHA256: {digest}")
    manifest = {
        "created_at_utc": utc_now(),
        "path": str(candidate.resolve()),
        "sha256": digest,
        "bytes": candidate.stat().st_size,
        "model_scale": scale,
        "parameter_count": parameter_count,
        "ultralytics_version": __import__("ultralytics").__version__,
    }
    write_json(ROOT / "models/pretrained/yolo11x.manifest.json", manifest)
    print(
        f"model={candidate.resolve()} scale={scale} parameters={parameter_count} "
        f"sha256={manifest['sha256']}"
    )


if __name__ == "__main__":
    main()
