from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank dual-protocol checkpoint evaluations.")
    parser.add_argument("--metrics", action="append", required=True)
    parser.add_argument("--output", default="runs/eval/candidate_ranking.csv")
    parser.add_argument("--allow-sample", action="store_true")
    args = parser.parse_args()

    rows = []
    for value in args.metrics:
        path = resolve_path(value)
        data = json.loads(path.read_text(encoding="utf-8"))
        is_full_validation = bool(data.get("is_full_validation")) and int(data.get("image_count", 0)) == 897
        if not is_full_validation and not args.allow_sample:
            raise ValueError(f"refusing to rank non-full validation metrics: {path}")
        rows.append(
            {
                "metrics_path": str(path),
                "image_count": int(data.get("image_count", 0)),
                "is_full_validation": is_full_validation,
                "passes_all_protocol_gates": bool(data["passes_all_protocol_gates"]),
                "passes_local_safety_target": bool(data["passes_local_safety_target"]),
                "worst_protocol_recall": float(data["worst_protocol_recall"]),
                "worst_protocol_fdr": float(data["worst_protocol_fdr"]),
                "selection_margin": float(data["selection_margin"]),
            }
        )
    rows.sort(
        key=lambda row: (
            row["is_full_validation"],
            row["passes_all_protocol_gates"],
            row["passes_local_safety_target"],
            row["selection_margin"],
            row["worst_protocol_recall"],
            -row["worst_protocol_fdr"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", *[key for key in rows[0] if key != "rank"]]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"candidates={len(rows)} output={output}")
    for row in rows:
        print(
            f"rank={row['rank']} pass={row['passes_all_protocol_gates']} "
            f"recall={row['worst_protocol_recall']:.6f} fdr={row['worst_protocol_fdr']:.6f} "
            f"metrics={row['metrics_path']}"
        )


if __name__ == "__main__":
    main()
