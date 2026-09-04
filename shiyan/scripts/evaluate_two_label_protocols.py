"""Evaluate one canonical prediction against original and revised label protocols."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.scripts.evaluate_official import _write_outputs, evaluate  # noqa: E402
from shiyan.src.inference.runner import read_image_list  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


def read_rename_manifest(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, str] = {}
    for row in rows:
        old_name = row["old_image_name"]
        new_name = row["new_image_name"]
        if old_name in mapping:
            raise ValueError(f"duplicate old image name in rename manifest: {old_name}")
        mapping[old_name] = new_name
    return mapping


def remap_result(
    document: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    remapped = copy.deepcopy(document)
    for entry in remapped["images"]:
        old_name = str(entry["file_name"])
        if old_name not in mapping:
            raise ValueError(f"prediction is missing from rename manifest: {old_name}")
        new_name = mapping[old_name]
        entry["file_name"] = new_name
        entry["image_id"] = Path(new_name).stem
    return remapped


def remap_timings(document: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    remapped = copy.deepcopy(document)
    for entry in remapped.get("images", []):
        old_name = str(entry["file_name"])
        if old_name not in mapping:
            raise ValueError(f"timing row is missing from rename manifest: {old_name}")
        entry["file_name"] = mapping[old_name]
    return remapped


def protocol_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_count": metrics["image_count"],
        "group_mean_recall": metrics["group_mean"]["recall"],
        "group_mean_fdr": metrics["group_mean"]["fdr"],
        "gates": metrics["gates"],
        "groups": {
            group: {
                "recall": values["recall"],
                "fdr": values["fdr"],
                "tp": values["tp"],
                "fp": values["fp"],
                "fn": values["fn"],
            }
            for group, values in metrics["groups"].items()
        },
        "timing": metrics["timing"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--timings", required=True)
    parser.add_argument("--original-image-list", required=True)
    parser.add_argument("--revised-image-list", required=True)
    parser.add_argument("--rename-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    timing_path = Path(args.timings)
    if not prediction_path.is_file() or not timing_path.is_file():
        raise FileNotFoundError("predictions or timings file is missing")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original_paths = read_image_list(args.original_image_list)
    original_names = [path.name for path in original_paths]
    document = json.loads(prediction_path.read_text(encoding="utf-8"))
    timing_document = json.loads(timing_path.read_text(encoding="utf-8"))
    validate_result_document(document, original_names)

    original_metrics = evaluate(
        prediction_path,
        Path(args.original_image_list),
        timing_path,
        {
            "label_version": "v0_original",
            "split_version": "v1_scene_80_20",
            "experiment_id": args.experiment_id,
            "run_id": f"{args.run_id}_d0",
        },
    )
    original_output = output_dir / "d0_original"
    original_output.mkdir(parents=True, exist_ok=True)
    (original_output / "source_files.json").write_text(
        json.dumps(
            {
                "predictions": str(prediction_path),
                "timings": str(timing_path),
                "image_list": str(Path(args.original_image_list)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_outputs(original_metrics, original_output / "metrics")

    mapping = read_rename_manifest(Path(args.rename_manifest))
    all_revised_paths = read_image_list(args.revised_image_list)
    revised_by_name = {path.name: path for path in all_revised_paths}
    mapped_names = [mapping[name] for name in original_names]
    missing = [name for name in mapped_names if name not in revised_by_name]
    if missing:
        raise ValueError(f"mapped image is absent from revised split: {missing[0]}")
    revised_paths = [revised_by_name[name] for name in mapped_names]

    remapped_document = remap_result(document, mapping)
    remapped_timing = remap_timings(timing_document, mapping)
    validate_result_document(remapped_document, mapped_names)
    revised_output = output_dir / "d3_manual"
    revised_output.mkdir(parents=True, exist_ok=True)
    revised_prediction_path = revised_output / "result.json"
    revised_timing_path = revised_output / "timings.json"
    revised_list_path = revised_output / "image_list.txt"
    revised_prediction_path.write_text(
        json.dumps(remapped_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    revised_timing_path.write_text(
        json.dumps(remapped_timing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    revised_list_path.write_text(
        "\n".join(str(path) for path in revised_paths) + "\n",
        encoding="utf-8",
    )
    revised_metrics = evaluate(
        revised_prediction_path,
        revised_list_path,
        revised_timing_path,
        {
            "label_version": "v2_data3_manual_revision_classfix",
            "split_version": "v1_scene_80_20_data3",
            "experiment_id": args.experiment_id,
            "run_id": f"{args.run_id}_d3",
        },
    )
    _write_outputs(revised_metrics, revised_output / "metrics")

    summaries = {
        "d0_original": protocol_summary(original_metrics),
        "d3_manual": protocol_summary(revised_metrics),
    }
    worst_recall = min(float(item["group_mean_recall"]) for item in summaries.values())
    worst_fdr = max(float(item["group_mean_fdr"]) for item in summaries.values())
    max_seconds = max(
        float(item["timing"]["max_image_seconds"])
        for item in summaries.values()
        if item["timing"].get("available")
    )
    robust_gates = {
        "recall_ge_0_85_on_both": worst_recall >= 0.85,
        "fdr_le_0_20_on_both": worst_fdr <= 0.20,
        "latency_le_20s": max_seconds <= 20.0,
    }
    comparison = {
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "prediction_source": str(prediction_path),
        "protocols": summaries,
        "robust_worst_case": {
            "group_mean_recall": worst_recall,
            "group_mean_fdr": worst_fdr,
            "max_image_seconds": max_seconds,
            "gates": robust_gates,
        },
    }
    (output_dir / "protocol_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Two-Label-Protocol Comparison",
        "",
        "These are internal validation results, not official hidden-test scores.",
        "",
        "| Protocol | Group-mean Recall | Group-mean FDR | Recall gate | FDR gate |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['group_mean_recall']:.6f} | {summary['group_mean_fdr']:.6f} | "
            f"{summary['gates']['recall_ge_0_85']} | {summary['gates']['fdr_le_0_20']} |"
        )
    lines.extend(
        [
            "",
            f"- Worst-case Recall: {worst_recall:.6f}",
            f"- Worst-case FDR: {worst_fdr:.6f}",
            f"- Maximum image time: {max_seconds:.6f} s",
            f"- Robust gates: {robust_gates}",
        ]
    )
    (output_dir / "protocol_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(robust_gates, ensure_ascii=False))
    print(f"comparison={output_dir / 'protocol_comparison.json'}")


if __name__ == "__main__":
    main()
