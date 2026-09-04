from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .geometry import tile_windows
from .labels import CLASS_COUNT, CLASS_NAMES
from .postprocess import apply_class_thresholds, class_aware_nms
from .types import Detection


class Predictor:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "0",
        imgsz: int = 1024,
        conf: float = 0.10,
        iou: float = 0.60,
        max_det_per_tile: int = 300,
        max_det_image: int = 3000,
        mode: str = "tiled",
        tile_size: int = 1024,
        tile_overlap: float = 0.20,
        merge_iou: float = 0.50,
        tile_batch: int = 1,
        half: bool = True,
        class_thresholds: Mapping[int, float] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {self.model_path}")
        if mode not in {"direct", "tiled"}:
            raise ValueError("mode must be direct or tiled")
        if min(imgsz, max_det_per_tile, max_det_image, tile_size, tile_batch) <= 0:
            raise ValueError("size, detection limits, and tile_batch must be positive")
        if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
            raise ValueError("conf and iou must be in [0, 1]")
        if not 0.0 <= tile_overlap < 0.9 or not 0.0 < merge_iou <= 1.0:
            raise ValueError("invalid tile overlap or merge IoU")
        self.device = str(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det_per_tile = max_det_per_tile
        self.max_det_image = max_det_image
        self.mode = mode
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.merge_iou = merge_iou
        self.tile_batch = tile_batch
        self.half = half
        self.class_thresholds = dict(class_thresholds or {})
        if self.class_thresholds and conf > min(self.class_thresholds.values()):
            raise ValueError("base conf cannot exceed the lowest class threshold")
        self.model = YOLO(str(self.model_path))
        self._validate_model_names()

    def _validate_model_names(self) -> None:
        names = self.model.names
        if isinstance(names, dict):
            try:
                normalized = [str(names[index]) for index in range(len(names))]
            except KeyError as exc:
                raise ValueError("model class mapping must use contiguous integer IDs") from exc
        else:
            normalized = [str(name) for name in names]
        if normalized != list(CLASS_NAMES):
            raise ValueError(
                "model must contain exactly the frozen 25 classes in order; "
                f"received {len(normalized)} classes: {normalized}"
            )

    def _predict_arrays(self, images: list[np.ndarray], *, max_det: int) -> list[list[Detection]]:
        results = self.model.predict(
            source=images,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            max_det=max_det,
            quantize=16 if self.half else 32,
            verbose=False,
        )
        return [self._read_result(result) for result in results]

    @staticmethod
    def _read_result(result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        detections: list[Detection] = []
        for xyxy, score, category in zip(coordinates, scores, classes):
            category_id = int(category)
            if not 0 <= category_id < CLASS_COUNT:
                raise ValueError(f"model returned category_id outside contract: {category_id}")
            x1, y1, x2, y2 = xyxy.tolist()
            detections.append(Detection(category_id, float(score), (x1, y1, x2, y2)))
        return detections

    def _predict_tiled(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        windows = tile_windows(width, height, self.tile_size, self.tile_overlap)
        detections: list[Detection] = []
        for start in range(0, len(windows), self.tile_batch):
            batch_windows = windows[start : start + self.tile_batch]
            tiles = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in batch_windows]
            batch_results = self._predict_arrays(tiles, max_det=self.max_det_per_tile)
            for result, (x1, y1, _x2, _y2) in zip(batch_results, batch_windows):
                for detection in result:
                    translated = (
                        detection.bbox[0] + x1,
                        detection.bbox[1] + y1,
                        detection.bbox[2] + x1,
                        detection.bbox[3] + y1,
                    )
                    clipped = (
                        max(0.0, min(float(width), translated[0])),
                        max(0.0, min(float(height), translated[1])),
                        max(0.0, min(float(width), translated[2])),
                        max(0.0, min(float(height), translated[3])),
                    )
                    if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                        detections.append(Detection(detection.category_id, detection.score, clipped))
        # Scores below the final class thresholds can never survive output and,
        # being lower-confidence, cannot suppress a retained box. Remove them
        # before cross-tile NMS to keep 10k-image merge cost bounded.
        filtered = apply_class_thresholds(detections, self.class_thresholds)
        return class_aware_nms(filtered, self.merge_iou, self.max_det_image)

    def predict_image(self, image_path: str | Path) -> tuple[int, int, list[Detection]]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read image: {image_path}")
        return self.predict_array(image)

    def predict_array(self, image: np.ndarray) -> tuple[int, int, list[Detection]]:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR array with shape HxWx3")
        height, width = image.shape[:2]
        if self.mode == "direct":
            detections = self._predict_arrays([image], max_det=self.max_det_image)[0]
            detections = apply_class_thresholds(detections, self.class_thresholds)
        else:
            detections = self._predict_tiled(image)
        return width, height, detections
