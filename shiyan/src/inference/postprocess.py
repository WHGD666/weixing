"""Post-processing shared by direct and tiled inference."""

from collections.abc import Iterable

from .types import Detection


def box_iou_xyxy(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    """Compute IoU for two pixel-coordinate boxes in xyxy format."""

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
    detections: Iterable[Detection],
    iou_threshold: float = 0.5,
    max_detections: int = 300,
) -> list[Detection]:
    """Merge tiled predictions while preserving different classes."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        suppressed = any(
            candidate.category_id == previous.category_id
            and box_iou_xyxy(candidate.bbox, previous.bbox) >= iou_threshold
            for previous in kept
        )
        if not suppressed:
            kept.append(candidate)
        if len(kept) >= max_detections:
            break
    return kept
