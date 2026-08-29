"""Apply selected class-specific score thresholds to an existing result.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shiyan.src.inference.filters import filter_class_thresholds  # noqa: E402
from shiyan.src.inference.runner import read_image_list  # noqa: E402
from shiyan.src.inference.schema import validate_result_document  # noqa: E402


def _parse_thresholds(values: list[str]) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for value in values:
        try:
            category_text, threshold_text = value.split("=", maxsplit=1)
            category_id = int(category_text)
            threshold = float(threshold_text)
        except ValueError as error:
            raise ValueError(f"invalid --class-threshold value: {value}; use CATEGORY_ID=THRESHOLD") from error
        if category_id in thresholds:
            raise ValueError(f"duplicate class threshold: {category_id}")
        thresholds[category_id] = threshold
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source official result.json")
    parser.add_argument("--image-list", required=True, help="image list corresponding to source result.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-threshold", action="append", default=[], help="repeatable CATEGORY_ID=THRESHOLD")
    parser.add_argument("--timings", help="optional timing sidecar to copy into output")
    args = parser.parse_args()

    image_paths = read_image_list(args.image_list)
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    expected_names = [path.name for path in image_paths]
    validate_result_document(source, expected_names)
    filtered = filter_class_thresholds(source, _parse_thresholds(args.class_threshold))
    validate_result_document(filtered, expected_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "image_list.txt").write_text(
        "\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8"
    )
    filter_config = {"source": str(Path(args.input)), "class_thresholds": _parse_thresholds(args.class_threshold)}
    (output_dir / "filter_config.json").write_text(
        json.dumps(filter_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.timings:
        (output_dir / "timings.json").write_text(
            Path(args.timings).read_text(encoding="utf-8"), encoding="utf-8"
        )
    source_count = sum(len(item["objects"]) for item in source["images"])
    filtered_count = sum(len(item["objects"]) for item in filtered["images"])
    print(f"source_objects={source_count}")
    print(f"filtered_objects={filtered_count}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
