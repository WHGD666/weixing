"""Reusable local inference components for the competition detector."""

from .labels import CLASS_NAMES
from .filters import filter_class_thresholds
from .types import Detection

__all__ = ["CLASS_NAMES", "Detection", "filter_class_thresholds"]
