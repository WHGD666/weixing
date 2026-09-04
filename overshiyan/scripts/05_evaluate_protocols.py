from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import resolve_path, utc_now, write_json
from src.data_contract import read_split_ids
from src.metrics import evaluate_protocol, write_protocol_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Score identical predictions against D3 and original D0 labels.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--timings")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-sample", action="store_true")
    args = parser.parse_args()

    predictions = resolve_path(args.predictions)
    timings = resolve_path(args.timings) if args.timings else None
    output = resolve_path(args.output_dir)
    val_ids = read_split_ids(ROOT / "data_registry/splits/v1_scene_80_20/val.txt")
    if args.allow_sample:
        import json

        document = json.loads(predictions.read_text(encoding="utf-8"))
        val_ids = [Path(entry["file_name"]).stem for entry in document["images"]]
    protocols = {
        "d3_manual_revision": ROOT / "workspace/data3_exp006/labels/val",
        "d0_original_mapped": ROOT / "data_registry/protocols/d0_original/labels/val",
    }
    results = {}
    for protocol_id, label_root in protocols.items():
        metrics = evaluate_protocol(
            prediction_path=predictions,
            label_root=label_root,
            expected_stems=val_ids,
            timings_path=timings,
            protocol_id=protocol_id,
        )
        results[protocol_id] = metrics
        write_protocol_outputs(metrics, output / protocol_id)

    recalls = [float(metrics["group_mean"]["recall"]) for metrics in results.values()]
    fdrs = [float(metrics["group_mean"]["fdr"]) for metrics in results.values()]
    all_gate_pass = all(all(value is True for value in metrics["gates"].values()) for metrics in results.values())
    comparison = {
        "created_at_utc": utc_now(),
        "scope": "dual_protocol_internal_validation",
        "prediction_path": str(predictions),
        "image_count": len(val_ids),
        "is_full_validation": len(val_ids) == 897,
        "protocols": {
            protocol_id: {
                "group_mean_recall": metrics["group_mean"]["recall"],
                "group_mean_fdr": metrics["group_mean"]["fdr"],
                "gates": metrics["gates"],
                "groups": metrics["groups"],
            }
            for protocol_id, metrics in results.items()
        },
        "worst_protocol_recall": min(recalls),
        "worst_protocol_fdr": max(fdrs),
        "passes_all_protocol_gates": all_gate_pass,
        "passes_local_safety_target": min(recalls) >= 0.90 and max(fdrs) <= 0.12,
        "selection_margin": min(min(recalls) - 0.85, 0.20 - max(fdrs)),
    }
    write_json(output / "protocol_comparison.json", comparison)
    print(
        f"all_gates={comparison['passes_all_protocol_gates']} "
        f"safety={comparison['passes_local_safety_target']} "
        f"worst_recall={comparison['worst_protocol_recall']:.6f} "
        f"worst_fdr={comparison['worst_protocol_fdr']:.6f}"
    )
    print(f"comparison={output / 'protocol_comparison.json'}")


if __name__ == "__main__":
    main()
