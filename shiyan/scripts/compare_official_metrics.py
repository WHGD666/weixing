#!/usr/bin/env python
"""Compare two local official-metric JSON files on the same evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


METRIC_PATHS = [
    ("overall.recall", True),
    ("overall.fdr", False),
    ("group_mean.recall", True),
    ("group_mean.fdr", False),
    ("three_group_macro.recall", True),
    ("three_group_macro.fdr", False),
    ("groups.ship.recall", True),
    ("groups.ship.fdr", False),
    ("groups.aircraft.recall", True),
    ("groups.aircraft.fdr", False),
    ("groups.vehicle.recall", True),
    ("groups.vehicle.fdr", False),
]


def _get(document: dict[str, Any], path: str) -> Any:
    value: Any = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _equal_when_present(left: Any, right: Any) -> bool:
    """Treat absent optional metadata as unknown, not as a mismatch."""

    return left is None or right is None or left == right


def _compatibility(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "image_count_equal": baseline.get("image_count") == candidate.get("image_count"),
        "metric_contract_equal": _equal_when_present(
            baseline.get("metric_contract"), candidate.get("metric_contract")
        ),
        "iou_thresholds_equal": _equal_when_present(
            baseline.get("iou_thresholds"), candidate.get("iou_thresholds")
        ),
    }
    baseline_metadata = baseline.get("evaluation_metadata", {})
    candidate_metadata = candidate.get("evaluation_metadata", {})
    for key in ("label_version", "split_version"):
        checks[f"{key}_equal"] = _equal_when_present(
            baseline_metadata.get(key), candidate_metadata.get(key)
        )
    return {"valid": all(checks.values()), "checks": checks}


def build_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path, higher_is_better in METRIC_PATHS:
        before = _get(baseline, path)
        after = _get(candidate, path)
        delta = None if before is None or after is None else float(after) - float(before)
        rows.append(
            {
                "metric": path,
                "baseline": before,
                "candidate": after,
                "delta": delta,
                "higher_is_better": higher_is_better,
            }
        )
    return {
        "comparison_scope": "same_local_evaluation_protocol_required",
        "baseline_metadata": baseline.get("evaluation_metadata", {}),
        "candidate_metadata": candidate.get("evaluation_metadata", {}),
        "baseline_image_count": baseline.get("image_count"),
        "candidate_image_count": candidate.get("image_count"),
        "compatibility": _compatibility(baseline, candidate),
        "rows": rows,
        "gate_candidates": {
            "baseline": baseline.get("gate_candidates", {}),
            "candidate": candidate.get("gate_candidates", {}),
        },
    }


def write_outputs(comparison: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["metric", "baseline", "candidate", "delta", "higher_is_better"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison["rows"])
    lines = [
        "# Local Metric Comparison",
        "",
        "This comparison is valid only when baseline and candidate use the same image list, labels, split and metric contract.",
        "",
        "| Metric | Baseline | Candidate | Delta | Direction |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparison["rows"]:
        direction = "higher" if row["higher_is_better"] else "lower"
        lines.append(
            f"| {row['metric']} | {row['baseline']} | {row['candidate']} | {row['delta']} | {direction} is better |"
        )
    lines.extend(
        [
            "",
            f"- Baseline images: {comparison['baseline_image_count']}",
            f"- Candidate images: {comparison['candidate_image_count']}",
            f"- Compatibility: {comparison['compatibility']}",
            f"- Baseline gate candidates: {comparison['gate_candidates']['baseline']}",
            f"- Candidate gate candidates: {comparison['gate_candidates']['candidate']}",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    comparison = build_comparison(baseline, candidate)
    write_outputs(comparison, Path(args.output_dir))
    result = {
        "ok": comparison["compatibility"]["valid"],
        "metrics": len(comparison["rows"]),
        "output": args.output_dir,
        "compatibility": comparison["compatibility"],
    }
    print(json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
