import unittest

from shiyan.src.inference.postprocess import class_aware_nms
from shiyan.src.inference.schema import validate_result_document
from shiyan.src.inference.types import Detection
from shiyan.src.inference.coco import result_to_coco
from shiyan.src.inference.filters import filter_class_thresholds
from shiyan.scripts.evaluate_official import _gate_flags, _score_components


class InferenceContractTests(unittest.TestCase):
    def test_class_aware_nms_keeps_different_classes(self) -> None:
        detections = [
            Detection(0, 0.9, (0.0, 0.0, 100.0, 100.0)),
            Detection(0, 0.8, (2.0, 2.0, 98.0, 98.0)),
            Detection(1, 0.7, (2.0, 2.0, 98.0, 98.0)),
        ]
        kept = class_aware_nms(detections, iou_threshold=0.5)
        self.assertEqual([(item.category_id, item.score) for item in kept], [(0, 0.9), (1, 0.7)])

    def test_official_result_schema(self) -> None:
        document = {
            "status": "success",
            "images": [
                {
                    "image_id": "sample",
                    "file_name": "sample.jpg",
                    "width": 100,
                    "height": 80,
                    "run_end_timestamp": 1,
                    "objects": [
                        {
                            "category_id": 24,
                            "category_name": "FSC",
                            "score": 0.5,
                            "bbox": [1.0, 2.0, 50.0, 60.0],
                        }
                    ],
                }
            ],
        }
        validate_result_document(document, ["sample.jpg"])

    def test_official_result_rejects_out_of_bounds_bbox(self) -> None:
        document = {
            "status": "success",
            "images": [
                {
                    "image_id": "sample",
                    "file_name": "sample.jpg",
                    "width": 100,
                    "height": 80,
                    "run_end_timestamp": 1,
                    "objects": [
                        {
                            "category_id": 0,
                            "category_name": "HM",
                            "score": 0.5,
                            "bbox": [0.1, 0.2, 101.0, 0.6],
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_result_document(document, ["sample.jpg"])

    def test_coco_conversion_changes_xyxy_to_xywh(self) -> None:
        document = {
            "status": "success",
            "images": [
                {
                    "image_id": "sample",
                    "file_name": "sample.jpg",
                    "width": 100,
                    "height": 80,
                    "run_end_timestamp": 1,
                    "objects": [
                        {
                            "category_id": 24,
                            "category_name": "FSC",
                            "score": 0.5,
                            "bbox": [10.0, 20.0, 50.0, 60.0],
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            result_to_coco(document),
            [{"image_id": "sample", "category_id": 24, "bbox": [10.0, 20.0, 40.0, 40.0], "score": 0.5}],
        )

    def test_class_threshold_filter_keeps_unspecified_classes(self) -> None:
        document = {
            "status": "success",
            "images": [
                {
                    "image_id": "sample",
                    "file_name": "sample.jpg",
                    "width": 100,
                    "height": 80,
                    "run_end_timestamp": 1,
                    "objects": [
                        {"category_id": 24, "category_name": "FSC", "score": 0.3, "bbox": [1, 2, 10, 20]},
                        {"category_id": 0, "category_name": "HM", "score": 0.3, "bbox": [20, 2, 30, 20]},
                    ],
                }
            ],
        }
        filtered = filter_class_thresholds(document, {24: 0.35})
        self.assertEqual(len(filtered["images"][0]["objects"]), 1)
        self.assertEqual(filtered["images"][0]["objects"][0]["category_id"], 0)

    def test_metric_gates_use_three_group_mean(self) -> None:
        passing = _gate_flags(0.85, 0.20, True)
        self.assertEqual(
            passing,
            {"recall_ge_0_85": True, "fdr_le_0_20": True, "latency_le_20s": True},
        )
        failing_group_mean = _gate_flags(0.849, 0.19, True)
        self.assertFalse(failing_group_mean["recall_ge_0_85"])
        failing_group_fdr = _gate_flags(0.86, 0.201, True)
        self.assertFalse(failing_group_fdr["fdr_le_0_20"])

    def test_published_score_formula(self) -> None:
        score = _score_components(0.944987666666667, 0.143308666666667, 1.837167)
        self.assertAlmostEqual(score["total_score"], 84.3313257, places=5)
        score_v2 = _score_components(0.942750666666667, 0.133293333333333, 1.791)
        self.assertAlmostEqual(score_v2["total_score"], 84.8309481, places=5)


if __name__ == "__main__":
    unittest.main()
