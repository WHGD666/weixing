"""Class-aware post-processing for cross-tile detections."""

from collections.abc import Iterable, Mapping


def box_iou_xyxy(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def class_aware_nms(
    detections: Iterable[dict[str, object]],
    iou_threshold: float,
    max_detections: int,
) -> list[dict[str, object]]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("merge_iou must be in (0, 1]")
    if max_detections <= 0:
        raise ValueError("max_det must be positive")

    ordered = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict[str, object]] = []
    for candidate in ordered:
        candidate_box = tuple(float(value) for value in candidate["bbox"])
        suppressed = any(
            int(candidate["category_id"]) == int(previous["category_id"])
            and box_iou_xyxy(candidate_box, tuple(float(value) for value in previous["bbox"])) >= iou_threshold
            for previous in kept
        )
        if not suppressed:
            kept.append(candidate)
        if len(kept) >= max_detections:
            break
    return kept


def filter_class_thresholds(
    detections: Iterable[dict[str, object]],
    thresholds: Mapping[int, float],
) -> list[dict[str, object]]:
    """Filter selected classes by score after all tile merging is complete."""

    normalized = {int(category_id): float(value) for category_id, value in thresholds.items()}
    for category_id, threshold in normalized.items():
        if category_id < 0:
            raise ValueError(f"category_id must be non-negative: {category_id}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]: {category_id}={threshold}")
    return [
        dict(item)
        for item in detections
        if float(item["score"]) >= normalized.get(int(item["category_id"]), 0.0)
    ]
