"""Small, framework-independent prediction types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    category_id: int
    score: float
    bbox: tuple[float, float, float, float]
