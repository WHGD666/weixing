"""Ultralytics-backed direct and tiled inference."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .labels import CLASS_COUNT, CLASS_NAMES
from .postprocess import class_aware_nms
from .types import Detection


class Predictor:
    """Load one frozen model and predict one image at a time."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "0",
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        mode: str = "direct",
        tile_size: int = 1024,
        tile_overlap: float = 0.2,
        merge_iou: float = 0.5,
        tile_batch: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {self.model_path}")
        if mode not in {"direct", "tiled"}:
            raise ValueError("mode must be direct or tiled")
        if imgsz <= 0 or max_det <= 0 or tile_size <= 0 or tile_batch <= 0:
            raise ValueError("imgsz, max_det, tile_size and tile_batch must be positive")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be in [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be in [0, 1]")
        if not 0.0 <= tile_overlap < 0.9:
            raise ValueError("tile_overlap must be in [0, 0.9)")

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
        if not 0.0 < merge_iou <= 1.0:
            raise ValueError("merge_iou must be in (0, 1]")
        self.model = YOLO(str(self.model_path))
        self._validate_model_names()

    def _validate_model_names(self) -> None:
        names = self.model.names
        if isinstance(names, dict):
            normalized = [names[index] for index in range(len(names))]
        else:
            normalized = list(names)
        if normalized[:CLASS_COUNT] != list(CLASS_NAMES):
            raise ValueError(
                "model class order does not match the frozen project label contract: "
                f"{normalized[:CLASS_COUNT]}"
            )

    def _predict_arrays(self, images: list[np.ndarray]) -> list[list[Detection]]:
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
    def _read_result(result: object, origin: tuple[int, int] = (0, 0)) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        offset_x, offset_y = origin
        detections: list[Detection] = []
        for coordinates, score, category in zip(xyxy, scores, classes):
            category_id = int(category)
            if not 0 <= category_id < CLASS_COUNT:
                raise ValueError(f"model returned category_id outside contract: {category_id}")
            x1, y1, x2, y2 = coordinates.tolist()
            detections.append(
                Detection(
                    category_id=category_id,
                    score=float(score),
                    bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                )
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

    def _predict_tiled(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        windows = self._tile_windows(width, height)
        detections: list[Detection] = []
        for start in range(0, len(windows), self.tile_batch):
            batch_windows = windows[start : start + self.tile_batch]
            tiles = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in batch_windows]
            batch_results = self._predict_arrays(tiles)
            for result, (x1, y1, x2, y2) in zip(batch_results, batch_windows):
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
                        detections.append(
                            Detection(detection.category_id, detection.score, clipped)
                        )
        return class_aware_nms(detections, self.merge_iou, self.max_det)

    def predict_image(self, image_path: str | Path) -> tuple[int, int, list[Detection]]:
        """Return width, height and pixel-coordinate detections for one image."""

        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read image: {path}")
        height, width = image.shape[:2]
        if self.mode == "direct":
            detections = self._predict_arrays([image])[0]
        else:
            detections = self._predict_tiled(image)
        return width, height, detections
