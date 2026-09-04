from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from .labels import CLASS_COUNT, CLASS_NAMES


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def validate_result_document(
    document: Mapping[str, object], expected_file_names: Iterable[str] | None = None
) -> None:
    if document.get("status") != "success":
        raise ValueError("result status must be success")
    entries = document.get("images")
    if not isinstance(entries, list):
        raise ValueError("images must be a list")
    expected = list(expected_file_names) if expected_file_names is not None else None
    actual_names: list[str] = []
    for image_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"images[{image_index}] must be an object")
        file_name = entry.get("file_name")
        image_id = entry.get("image_id")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"images[{image_index}].file_name must be non-empty")
        if not isinstance(image_id, str) or image_id != Path(file_name).stem:
            raise ValueError(f"image_id does not match file_name for {file_name}")
        if file_name in actual_names:
            raise ValueError(f"duplicate file_name: {file_name}")
        actual_names.append(file_name)
        width = _number(entry.get("width"), f"width for {file_name}")
        height = _number(entry.get("height"), f"height for {file_name}")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions for {file_name}")
        timestamp = entry.get("run_end_timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError(f"invalid run_end_timestamp for {file_name}")
        objects = entry.get("objects")
        if not isinstance(objects, list):
            raise ValueError(f"objects must be a list for {file_name}")
        for object_index, item in enumerate(objects):
            if not isinstance(item, Mapping):
                raise ValueError(f"objects[{object_index}] must be an object for {file_name}")
            category_id = item.get("category_id")
            if isinstance(category_id, bool) or not isinstance(category_id, int):
                raise ValueError(f"category_id must be an integer for {file_name}")
            if not 0 <= category_id < CLASS_COUNT:
                raise ValueError(f"category_id out of range for {file_name}: {category_id}")
            if item.get("category_name") != CLASS_NAMES[category_id]:
                raise ValueError(f"category_name mismatch for {file_name}, id={category_id}")
            score = _number(item.get("score"), f"score for {file_name}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"score outside [0, 1] for {file_name}")
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"bbox must be [x1, y1, x2, y2] for {file_name}")
            x1, y1, x2, y2 = [_number(value, f"bbox for {file_name}") for value in bbox]
            if not (0.0 <= x1 <= x2 <= width and 0.0 <= y1 <= y2 <= height):
                raise ValueError(f"bbox outside image bounds for {file_name}: {bbox}")
    if expected is not None and actual_names != expected:
        raise ValueError(
            f"result order/coverage mismatch: expected {len(expected)}, got {len(actual_names)}"
        )
