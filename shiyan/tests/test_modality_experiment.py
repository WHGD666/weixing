import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from shiyan.scripts.run_v2_modality_experiment import class_threshold, filter_detections, image_modality


class ModalityExperimentTests(unittest.TestCase):
    def test_grayscale_image_is_classified_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.jpg"
            image = np.full((20, 20), 80, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            stats = image_modality(
                path,
                gray_mean_range_threshold=0.02,
                gray_pixel_range_threshold=0.04,
                gray_pixel_fraction_threshold=0.05,
            )
            self.assertEqual(stats["modality"], "grayscale")

    def test_color_ship_policy_can_drop_only_ship_classes(self) -> None:
        detections = [
            {"category_id": 0, "score": 0.95, "bbox": [0, 0, 10, 10]},
            {"category_id": 4, "score": 0.95, "bbox": [0, 0, 10, 10]},
            {"category_id": 24, "score": 0.95, "bbox": [0, 0, 10, 10]},
        ]
        kept = filter_detections(
            detections,
            modality="color",
            ship_gray_conf=0.20,
            ship_color_conf=0.60,
            color_ship_policy="drop",
            aircraft_conf=0.30,
            fsc_conf=0.35,
        )
        self.assertEqual([item["category_id"] for item in kept], [4, 24])

    def test_grayscale_ship_uses_lower_threshold(self) -> None:
        self.assertEqual(
            class_threshold(
                0,
                modality="grayscale",
                ship_gray_conf=0.20,
                ship_color_conf=0.60,
                color_ship_policy="threshold",
                aircraft_conf=0.30,
                fsc_conf=0.35,
            ),
            0.20,
        )


if __name__ == "__main__":
    unittest.main()
