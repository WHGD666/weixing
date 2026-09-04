import unittest

from shiyan.src.inference.algorithmic_postprocess import (
    apply_modality_and_thresholds,
    combine_models,
    combine_multiple_models,
    group_aware_nms,
)


def detection(category_id: int, score: float, bbox: list[float]) -> dict[str, object]:
    return {"category_id": category_id, "score": score, "bbox": bbox}


class AlgorithmicPostprocessTests(unittest.TestCase):
    def test_group_nms_suppresses_different_ship_subclasses(self) -> None:
        combined = combine_models(
            [
                detection(0, 0.90, [0, 0, 100, 100]),
                detection(1, 0.80, [2, 2, 98, 98]),
                detection(4, 0.70, [2, 2, 98, 98]),
            ],
            [],
            fusion_mode="source-a",
            consensus_iou=0.5,
            source_preference={},
        )

        kept = group_aware_nms(combined, iou_threshold=0.5)

        self.assertEqual([item["category_id"] for item in kept], [0, 4])

    def test_group_nms_does_not_let_support_override_higher_score(self) -> None:
        candidates = [
            {
                "category_id": 0,
                "score": 0.40,
                "bbox": [0, 0, 100, 100],
                "group": "ship",
                "sources": ("a", "b"),
                "support": 2,
            },
            {
                "category_id": 1,
                "score": 0.90,
                "bbox": [2, 2, 98, 98],
                "group": "ship",
                "sources": ("a",),
                "support": 1,
            },
        ]

        kept = group_aware_nms(candidates, iou_threshold=0.5)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["category_id"], 1)

    def test_route_uses_declared_source_for_each_group(self) -> None:
        model_a = [
            detection(0, 0.90, [0, 0, 10, 10]),
            detection(4, 0.80, [20, 20, 30, 30]),
            detection(24, 0.70, [40, 40, 50, 50]),
        ]
        model_b = [
            detection(0, 0.60, [1, 1, 11, 11]),
            detection(4, 0.50, [21, 21, 31, 31]),
            detection(24, 0.95, [41, 41, 51, 51]),
        ]

        combined = combine_models(
            model_a,
            model_b,
            fusion_mode="route",
            consensus_iou=0.5,
            source_preference={"ship": "a", "aircraft": "a", "vehicle": "b"},
        )

        self.assertEqual([item["score"] for item in combined], [0.90, 0.80, 0.95])
        self.assertEqual([item["sources"] for item in combined], [("a",), ("a",), ("b",)])

    def test_consensus_fuses_same_group_boxes(self) -> None:
        combined = combine_models(
            [detection(0, 0.80, [0, 0, 100, 100])],
            [detection(1, 0.90, [5, 5, 105, 105])],
            fusion_mode="consensus",
            consensus_iou=0.5,
            source_preference={"ship": "a"},
        )

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["category_id"], 0)
        self.assertEqual(combined[0]["support"], 2)
        self.assertEqual(combined[0]["sources"], ("a", "b"))
        self.assertAlmostEqual(combined[0]["score"], 0.90)

    def test_strict_modality_is_bidirectional(self) -> None:
        combined = combine_models(
            [
                detection(0, 0.90, [0, 0, 10, 10]),
                detection(4, 0.90, [20, 20, 30, 30]),
                detection(24, 0.90, [40, 40, 50, 50]),
            ],
            [],
            fusion_mode="source-a",
            consensus_iou=0.5,
            source_preference={},
        )
        thresholds = {"ship": 0.0, "aircraft": 0.0, "vehicle": 0.0}

        color = apply_modality_and_thresholds(
            combined,
            modality="color",
            modality_policy="strict",
            consensus_thresholds=thresholds,
            single_thresholds=thresholds,
            ship_color_conf=0.6,
            nonship_gray_conf=0.6,
        )
        grayscale = apply_modality_and_thresholds(
            combined,
            modality="grayscale",
            modality_policy="strict",
            consensus_thresholds=thresholds,
            single_thresholds=thresholds,
            ship_color_conf=0.6,
            nonship_gray_conf=0.6,
        )

        self.assertEqual([item["group"] for item in color], ["aircraft", "vehicle"])
        self.assertEqual([item["group"] for item in grayscale], ["ship"])

    def test_consensus_can_use_lower_threshold_than_single_model(self) -> None:
        combined = combine_models(
            [
                detection(24, 0.30, [0, 0, 10, 10]),
                detection(24, 0.30, [30, 30, 40, 40]),
            ],
            [detection(24, 0.32, [1, 1, 11, 11])],
            fusion_mode="consensus",
            consensus_iou=0.5,
            source_preference={"vehicle": "b"},
        )

        kept = apply_modality_and_thresholds(
            combined,
            modality="color",
            modality_policy="off",
            consensus_thresholds={"ship": 0.2, "aircraft": 0.2, "vehicle": 0.25},
            single_thresholds={"ship": 0.4, "aircraft": 0.4, "vehicle": 0.45},
            ship_color_conf=0.6,
            nonship_gray_conf=0.6,
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["support"], 2)

    def test_three_model_consensus_records_three_independent_votes(self) -> None:
        combined = combine_multiple_models(
            {
                "a": [detection(24, 0.80, [0, 0, 20, 20])],
                "b": [detection(24, 0.70, [1, 1, 21, 21])],
                "c": [detection(24, 0.90, [2, 2, 22, 22])],
            },
            fusion_mode="consensus",
            consensus_iou=0.5,
            source_preference={"ship": "a", "aircraft": "a", "vehicle": "c"},
        )

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["support"], 3)
        self.assertEqual(combined[0]["category_id"], 24)
        self.assertEqual(set(combined[0]["sources"]), {"a", "b", "c"})

    def test_one_model_cannot_cast_two_votes_in_one_cluster(self) -> None:
        combined = combine_multiple_models(
            {
                "a": [
                    detection(0, 0.90, [0, 0, 20, 20]),
                    detection(1, 0.80, [1, 1, 21, 21]),
                ],
                "b": [detection(0, 0.70, [2, 2, 22, 22])],
            },
            fusion_mode="consensus",
            consensus_iou=0.5,
            source_preference={"ship": "a", "aircraft": "a", "vehicle": "a"},
        )

        self.assertEqual(sorted(item["support"] for item in combined), [1, 2])

    def test_vote_mode_drops_single_model_only_detection(self) -> None:
        combined = combine_multiple_models(
            {
                "a": [
                    detection(4, 0.90, [0, 0, 20, 20]),
                    detection(4, 0.90, [50, 50, 70, 70]),
                ],
                "b": [detection(4, 0.80, [1, 1, 21, 21])],
                "c": [],
            },
            fusion_mode="vote",
            consensus_iou=0.5,
            source_preference={"ship": "a", "aircraft": "a", "vehicle": "a"},
            minimum_support=2,
        )

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["support"], 2)


if __name__ == "__main__":
    unittest.main()
