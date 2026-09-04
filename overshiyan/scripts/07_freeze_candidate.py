from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import resolve_path, sha256_file, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an evaluated checkpoint for 3090 packaging.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("name may contain only letters, numbers, underscore, and hyphen")
    model = resolve_path(args.model)
    metrics_path = resolve_path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics.get("is_full_validation") or int(metrics.get("image_count", 0)) != 897:
        raise ValueError("candidate freezing requires the complete fixed 897-image validation set")
    if not metrics.get("passes_all_protocol_gates") and not args.allow_gate_failure:
        raise ValueError("candidate fails at least one protocol gate; use --allow-gate-failure only deliberately")

    frozen_model = ROOT / "models/frozen" / f"{args.name}.pt"
    if frozen_model.exists():
        raise FileExistsError(f"frozen model already exists: {frozen_model}")
    if args.dry_run:
        print(
            f"dry_run=passed model={model} model_sha256={sha256_file(model)} "
            f"metrics={metrics_path} destination={frozen_model}"
        )
        return
    frozen_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, frozen_model)
    submit_model = ROOT / "submit/models/best.pt"
    submit_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, submit_model)
    digest = sha256_file(frozen_model)
    manifest = {
        "name": args.name,
        "frozen_at_utc": utc_now(),
        "source_model": str(model),
        "source_model_sha256": sha256_file(model),
        "frozen_model": str(frozen_model),
        "frozen_model_sha256": digest,
        "submit_model": str(submit_model),
        "submit_model_sha256": sha256_file(submit_model),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metric_summary": metrics,
    }
    manifest_path = frozen_model.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    write_json(ROOT / "submit/model_manifest.json", manifest)
    print(f"frozen={frozen_model} sha256={digest}")
    print(f"submit_model={submit_model} manifest={manifest_path}")


if __name__ == "__main__":
    main()
