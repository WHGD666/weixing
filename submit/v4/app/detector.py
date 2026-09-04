"""Ultralytics detector used by the v2 delivery entrypoint."""

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from labels import CLASS_COUNT, CLASS_NAMES
from postprocess import class_aware_nms, filter_class_thresholds


class Detector:
    """Load one model and expose direct or tiled pixel-coordinate inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "0",
        imgsz: int = 1024,
        conf: float = 0.30,
        iou: float = 0.60,
        max_det: int = 300,
        mode: str = "tiled",
        tile_size: int = 1024,
        tile_overlap: float = 0.20,
        merge_iou: float = 0.50,
        tile_batch: int = 4,
        class_thresholds: Mapping[int, float] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {self.model_path}")
        if mode not in {"direct", "tiled"}:
            raise ValueError("mode must be direct or tiled")
        if imgsz <= 0 or max_det <= 0 or tile_size <= 0 or tile_batch <= 0:
            raise ValueError("imgsz, max_det, tile_size and tile_batch must be positive")
        if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
            raise ValueError("conf and iou must be in [0, 1]")
        if not 0.0 <= tile_overlap < 0.9:
            raise ValueError("tile_overlap must be in [0, 0.9)")
        if not 0.0 < merge_iou <= 1.0:
            raise ValueError("merge_iou must be in (0, 1]")

        self.device = str(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.mode = mode
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.merge_iou = merge_iou
        self.tile_batch = tile_batch
        self.class_thresholds = {int(category_id): float(value) for category_id, value in (class_thresholds or {}).items()}
        for category_id, threshold in self.class_thresholds.items():
            if not 0 <= category_id < CLASS_COUNT:
                raise ValueError(f"class threshold category_id outside contract: {category_id}")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"class threshold must be in [0, 1]: {category_id}={threshold}")
        self.model = YOLO(str(self.model_path))
        self._validate_model_names()

    def _validate_model_names(self) -> None:
        names = self.model.names
        if isinstance(names, dict):
            normalized = [names[index] for index in range(len(names))]
        else:
            normalized = list(names)
        if normalized[:CLASS_COUNT] != list(CLASS_NAMES):
            raise ValueError(f"model class order does not match frozen label contract: {normalized[:CLASS_COUNT]}")

    def _predict_arrays(self, images: list[np.ndarray]) -> list[list[dict[str, object]]]:
        results = self.model.predict(
            source=images,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )
        return [self._read_result(result) for result in results]

    @staticmethod
    def _read_result(result: object) -> list[dict[str, object]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        detections: list[dict[str, object]] = []
        for coordinates, score, category in zip(xyxy, scores, classes):
            category_id = int(category)
            if not 0 <= category_id < CLASS_COUNT:
                raise ValueError(f"model returned category_id outside contract: {category_id}")
            detections.append(
                {
                    "category_id": category_id,
                    "score": float(score),
                    "bbox": [float(value) for value in coordinates.tolist()],
                }
            )
        return detections

    def _tile_windows(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        tile = min(self.tile_size, width, height)
        if tile <= 0:
            raise ValueError("image dimensions must be positive")
        step = max(1, int(round(tile * (1.0 - self.tile_overlap))))

        def starts(length: int) -> list[int]:
            if length <= tile:
                return [0]
            values = list(range(0, length - tile + 1, step))
            last = length - tile
            if values[-1] != last:
                values.append(last)
            return values

        return [
            (x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height)
            for x in starts(width)
        ]

    def _predict_tiled(self, image: np.ndarray) -> list[dict[str, object]]:
        height, width = image.shape[:2]
        windows = self._tile_windows(width, height)
        detections: list[dict[str, object]] = []
        for start in range(0, len(windows), self.tile_batch):
            batch_windows = windows[start : start + self.tile_batch]
            tiles = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in batch_windows]
            batch_results = self._predict_arrays(tiles)
            for result, (x1, y1, x2, y2) in zip(batch_results, batch_windows):
                for detection in result:
                    box = detection["bbox"]
                    translated = [
                        float(box[0]) + x1,
                        float(box[1]) + y1,
                        float(box[2]) + x1,
                        float(box[3]) + y1,
                    ]
                    clipped = [
                        max(0.0, min(float(width), translated[0])),
                        max(0.0, min(float(height), translated[1])),
                        max(0.0, min(float(width), translated[2])),
                        max(0.0, min(float(height), translated[3])),
                    ]
                    if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                        detections.append(
                            {
                                "category_id": int(detection["category_id"]),
                                "score": float(detection["score"]),
                                "bbox": clipped,
                            }
                        )
        return class_aware_nms(detections, self.merge_iou, self.max_det)

    def predict_image(self, image_path: str | Path) -> tuple[int, int, list[dict[str, object]]]:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read image: {path}")
        height, width = image.shape[:2]
        if self.mode == "direct":
            detections = self._predict_arrays([image])[0]
        else:
            detections = self._predict_tiled(image)
        return width, height, filter_class_thresholds(detections, self.class_thresholds)
