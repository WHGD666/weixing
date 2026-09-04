from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.metrics import evaluate_protocol


class MetricTests(unittest.TestCase):
    def test_exact_match_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels"
            labels.mkdir()
            (labels / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            result = {
                "status": "success",
                "images": [
                    {
                        "image_id": "0001",
                        "file_name": "0001.jpg",
                        "width": 100,
                        "height": 100,
                        "run_end_timestamp": 1,
                        "objects": [
                            {
                                "category_id": 0,
                                "category_name": "HM",
                                "score": 0.9,
                                "bbox": [40.0, 40.0, 60.0, 60.0],
                            }
                        ],
                    }
                ],
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            metrics = evaluate_protocol(
                prediction_path=result_path,
                label_root=labels,
                expected_stems=["0001"],
                timings_path=None,
                protocol_id="synthetic",
            )
            self.assertEqual(metrics["overall"]["tp"], 1)
            self.assertEqual(metrics["overall"]["fp"], 0)
            self.assertEqual(metrics["overall"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
