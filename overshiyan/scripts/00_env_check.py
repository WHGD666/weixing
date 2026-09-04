from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import environment_snapshot, write_json


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the fixed training or deployment environment.")
    parser.add_argument("--target", choices=("train5090", "deploy3090"), required=True)
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--output", default="runs/preflight/environment.json")
    args = parser.parse_args()

    snapshot = environment_snapshot()
    free_bytes = shutil.disk_usage(ROOT).free
    snapshot["disk_free_bytes"] = free_bytes
    snapshot["target"] = args.target
    errors: list[str] = []

    if snapshot.get("ultralytics") != "8.4.128":
        errors.append(f"ultralytics must be 8.4.128, got {snapshot.get('ultralytics')}")
    if not snapshot.get("cuda_available"):
        errors.append("CUDA is not available to PyTorch")
    gpu_name = str(snapshot.get("gpu_name", ""))
    if not args.allow_other_gpu:
        required_name = "5090" if args.target == "train5090" else "3090"
        if required_name not in gpu_name:
            errors.append(f"expected an RTX {required_name}, got {gpu_name or 'unknown GPU'}")
    capability = tuple(snapshot.get("gpu_capability", []))
    if args.target == "train5090":
        if sys.version_info[:2] != (3, 12):
            errors.append(f"training image must use Python 3.12, got {sys.version.split()[0]}")
        if _version_tuple(str(snapshot.get("torch"))) < (2, 10):
            errors.append(f"training torch must be >=2.10, got {snapshot.get('torch')}")
        if _version_tuple(str(snapshot.get("torch_cuda"))) < (12, 8):
            errors.append(f"training CUDA runtime must be >=12.8, got {snapshot.get('torch_cuda')}")
        if capability and capability < (12, 0):
            errors.append(f"unexpected 5090 compute capability {capability}")
        if free_bytes < 60 * 1024**3:
            errors.append(f"less than 60 GiB free disk: {free_bytes / 1024**3:.1f} GiB")
    else:
        if _version_tuple(str(snapshot.get("torch")))[:2] != (2, 5):
            errors.append(f"deployment torch must be 2.5.x, got {snapshot.get('torch')}")
        memory = int(snapshot.get("gpu_memory_bytes", 0))
        if memory and memory < 22 * 1024**3:
            errors.append(f"deployment GPU exposes only {memory / 1024**3:.1f} GiB")

    snapshot["ok"] = not errors
    snapshot["errors"] = errors
    output = (ROOT / args.output).resolve()
    write_json(output, snapshot)
    print(f"ok={snapshot['ok']} target={args.target} gpu={gpu_name!r} output={output}")
    for error in errors:
        print(f"ERROR: {error}")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
