from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    environment_snapshot,
    load_yaml,
    resolve_path,
    sha256_file,
    utc_now,
    write_json,
)


EXPECTED_ULTRALYTICS = "8.4.128"


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    result = []
    for token in value.split("+")[0].split("."):
        if not token.isdigit():
            break
        result.append(int(token))
    return tuple(result)


def _preflight(config_path: Path, config: dict[str, object]) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    errors: list[str] = []
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        errors.append(f"ultralytics must be {EXPECTED_ULTRALYTICS}, got {ultralytics.__version__}")
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "5090" not in gpu_name:
        errors.append(f"EXP006 is fixed to RTX 5090 training, got {gpu_name or 'unknown GPU'}")
    if sys.version_info[:2] != (3, 12):
        errors.append(f"EXP006 requires Python 3.12, got {sys.version.split()[0]}")
    if _version_tuple(torch.__version__) < (2, 10):
        errors.append(f"EXP006 requires torch >=2.10, got {torch.__version__}")
    if _version_tuple(torch.version.cuda) < (12, 8):
        errors.append(f"EXP006 requires CUDA runtime >=12.8, got {torch.version.cuda}")
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (12, 0):
        errors.append(f"unexpected 5090 compute capability: {torch.cuda.get_device_capability(0)}")
    if shutil.disk_usage(ROOT).free < 60 * 1024**3:
        errors.append("less than 60 GiB free disk")

    model_path = resolve_path(str(config["model"]))
    data_path = resolve_path(str(config["data"]))
    prepared_path = data_path.parent / "prepared_dataset.json"
    audit_path = ROOT / "data_registry/audits/data3_source_audit.json"
    if model_path.name != "yolo11x.pt":
        errors.append(f"expected official yolo11x.pt, got {model_path.name}")
    for required in (model_path, data_path, prepared_path, audit_path):
        if not required.is_file():
            errors.append(f"required file missing: {required}")
    unknown_args = sorted(
        set(config) - set(DEFAULT_CFG_DICT) - {"experiment_id", "wall_clock_limit_hours", "model"}
    )
    if unknown_args:
        errors.append(f"unsupported Ultralytics configuration keys: {unknown_args}")
    if int(config.get("epochs", 0)) != 60 or int(config.get("imgsz", 0)) != 1024:
        errors.append("EXP006 fixed schedule must remain epochs=60 and imgsz=1024")
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("ok"):
            errors.append(f"source audit is not clean: {audit_path}")
        if not all(audit.get("source_class_contract", {}).values()):
            errors.append("source Data3 class contract was not verified")
    contract = load_yaml(ROOT / "data_registry/contracts/task_contract.yaml")
    if model_path.is_file() and sha256_file(model_path) != str(contract["pretrained_model_sha256"]):
        errors.append("pretrained yolo11x.pt SHA256 does not match the frozen experiment contract")
    if prepared_path.is_file() and audit_path.is_file():
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if prepared.get("source_fingerprint_after") != audit.get("fingerprint_sha256"):
            errors.append("prepared view was not built from the currently audited Data3 source")
    if data_path.is_file():
        data_config = load_yaml(data_path)
        names = data_config.get("names")
        if isinstance(names, dict):
            normalized_names = [str(names[index]) for index in range(len(names))]
        elif isinstance(names, list):
            normalized_names = [str(value) for value in names]
        else:
            normalized_names = []
        from src.inference.labels import CLASS_NAMES

        if normalized_names != list(CLASS_NAMES):
            errors.append("prepared data.yaml does not contain the exact frozen class order")
    if model_path.is_file():
        try:
            pretrained = YOLO(str(model_path))
            scale = pretrained.model.yaml.get("scale")
            parameter_count = sum(parameter.numel() for parameter in pretrained.model.parameters())
            if scale != "x" or parameter_count < 50_000_000:
                errors.append(
                    f"pretrained checkpoint is not YOLO11x: scale={scale!r}, parameters={parameter_count}"
                )
        except Exception as exc:
            errors.append(f"unable to load pretrained checkpoint: {exc}")

    result: dict[str, object] = {
        "ok": not errors,
        "checked_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path) if model_path.is_file() else None,
        "data": str(data_path),
        "prepared_dataset_sha256": sha256_file(prepared_path) if prepared_path.is_file() else None,
        "environment": environment_snapshot(),
        "errors": errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed EXP006-X0 YOLO11x training experiment.")
    parser.add_argument("--config", default="configs/train/exp006_x0_data3_yolo11x_1024.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", help="Resume only an interrupted Ultralytics last.pt run.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    experiment_id = str(config.get("experiment_id", ""))
    if experiment_id != "EXP006_X0_data3_yolo11x_1024":
        raise ValueError(f"Unexpected experiment_id in fixed entry point: {experiment_id}")
    preflight = _preflight(config_path, config)
    preflight_path = ROOT / "runs/preflight/EXP006_X0_preflight.json"
    write_json(preflight_path, preflight)
    print(f"preflight_ok={preflight['ok']} output={preflight_path}")
    for error in preflight["errors"]:
        print(f"ERROR: {error}")
    if not preflight["ok"]:
        raise SystemExit(2)
    if args.preflight_only:
        return

    import torch
    from ultralytics import YOLO

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_args = dict(config)
    train_args.pop("experiment_id", None)
    wall_clock_limit_hours = float(train_args.pop("wall_clock_limit_hours"))
    if not 0.5 <= wall_clock_limit_hours <= 4.0:
        raise ValueError("wall_clock_limit_hours must be between 0.5 and 4.0")
    model_path = resolve_path(str(train_args.pop("model")))
    train_args["data"] = str(resolve_path(str(train_args["data"])))
    train_args["project"] = str(resolve_path(str(train_args["project"])))
    run_dir = Path(str(train_args["project"])) / str(train_args["name"])
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"fresh EXP006 run directory is not empty: {run_dir}; resume last.pt or create a new experiment ID"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "frozen_train_config.yaml")

    manifest: dict[str, object] = {
        "experiment_id": experiment_id,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "pretrained_model": str(model_path),
        "pretrained_model_sha256": sha256_file(model_path),
        "prepared_dataset_sha256": preflight["prepared_dataset_sha256"],
        "environment": preflight["environment"],
        "resume_checkpoint": str(resolve_path(args.resume)) if args.resume else None,
        "wall_clock_limit_hours": wall_clock_limit_hours,
    }
    manifest_path = run_dir / "run_manifest.json"
    previous_manifest = None
    if args.resume and manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["previous_manifest"] = previous_manifest
    write_json(manifest_path, manifest)

    try:
        training_started = time.monotonic()
        epoch_started = training_started
        timing_path = run_dir / "epoch_timings.csv"

        def mark_epoch_start(_trainer: object) -> None:
            nonlocal epoch_started
            epoch_started = time.monotonic()

        def stop_at_wall_clock_limit(trainer: object) -> None:
            elapsed_hours = (time.monotonic() - training_started) / 3600.0
            if elapsed_hours >= wall_clock_limit_hours:
                setattr(trainer, "stop", True)
                print(
                    f"wall_clock_limit_reached={elapsed_hours:.3f}h; "
                    "finishing validation and checkpoint save",
                    flush=True,
                )

        def record_epoch_timing(trainer: object) -> None:
            now = time.monotonic()
            epoch_seconds = now - epoch_started
            total_seconds = now - training_started
            completed_epochs = int(getattr(trainer, "epoch")) + 1
            configured_epochs = int(config["epochs"])
            projected_seconds = total_seconds / max(1, completed_epochs) * configured_epochs
            file_exists = timing_path.is_file()
            with timing_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "epoch",
                        "epoch_seconds_including_validation",
                        "invocation_total_seconds",
                        "projected_60_epoch_hours",
                    ),
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "epoch": completed_epochs,
                        "epoch_seconds_including_validation": round(epoch_seconds, 3),
                        "invocation_total_seconds": round(total_seconds, 3),
                        "projected_60_epoch_hours": round(projected_seconds / 3600.0, 3),
                    }
                )
            print(
                f"runtime_projection epoch={completed_epochs} epoch_seconds={epoch_seconds:.1f} "
                f"projected_60_epochs={projected_seconds / 3600.0:.2f}h",
                flush=True,
            )

        def add_callbacks(model: object) -> None:
            model.add_callback("on_train_epoch_start", mark_epoch_start)
            model.add_callback("on_train_epoch_end", stop_at_wall_clock_limit)
            model.add_callback("on_fit_epoch_end", record_epoch_timing)

        if args.resume:
            resume_path = resolve_path(args.resume)
            if resume_path.name != "last.pt" or not resume_path.is_file():
                raise ValueError("--resume must point to an existing last.pt")
            if resume_path.parent.parent.resolve() != run_dir.resolve():
                raise ValueError(f"resume checkpoint must belong to the fixed EXP006 run: {run_dir}")
            model = YOLO(str(resume_path))
            add_callbacks(model)
            model.train(resume=True)
        else:
            model = YOLO(str(model_path))
            add_callbacks(model)
            model.train(**train_args)
        weights_dir = run_dir / "weights"
        manifest["status"] = "COMPLETED"
        manifest["finished_at_utc"] = utc_now()
        manifest["weights"] = {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name in ("best.pt", "last.pt")
            if (path := weights_dir / name).is_file()
        }
        best_path = weights_dir / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"training completed without best.pt: {best_path}")
        best_names = YOLO(str(best_path)).names
        if isinstance(best_names, dict):
            normalized_names = [str(best_names[index]) for index in range(len(best_names))]
        else:
            normalized_names = [str(value) for value in best_names]
        from src.inference.labels import CLASS_NAMES

        if normalized_names != list(CLASS_NAMES):
            raise ValueError("trained checkpoint does not contain the exact frozen 25-class order")
        manifest["class_names_verified"] = True
        write_json(manifest_path, manifest)
        print(f"status=COMPLETED run={run_dir} manifest={manifest_path}")
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["finished_at_utc"] = utc_now()
        manifest["error"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
