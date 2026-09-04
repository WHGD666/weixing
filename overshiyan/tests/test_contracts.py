from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

from src.common import sha256_file
from src.data_contract import deduplicated_label_text, parse_label_file
from src.data_contract import load_class_names, read_split_ids
from src.inference.labels import CLASS_NAMES
from src.inference.geometry import tile_windows
from src.inference.postprocess import apply_class_thresholds, class_aware_nms
from src.inference.schema import validate_result_document
from src.inference.types import Detection


class LabelContractTests(unittest.TestCase):
    def test_exact_duplicate_rows_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            text, source_rows, duplicates = deduplicated_label_text(path, len(CLASS_NAMES))
            self.assertEqual(source_rows, 2)
            self.assertEqual(duplicates, 1)
            self.assertEqual(text.count("\n"), 1)

    def test_invalid_class_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("25 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            _rows, errors = parse_label_file(path, len(CLASS_NAMES))
            self.assertTrue(errors)


class InferenceContractTests(unittest.TestCase):
    def test_elongated_image_uses_rectangular_tiles(self) -> None:
        windows = tile_windows(4096, 512, 1024, 0.20)
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(y1 == 0 and y2 == 512 for _x1, y1, _x2, y2 in windows))
        self.assertTrue(all(x2 - x1 == 1024 for x1, _y1, x2, _y2 in windows))

    def test_windows_cover_final_edges(self) -> None:
        windows = tile_windows(2500, 2100, 1024, 0.20)
        self.assertTrue(any(x2 == 2500 for _x1, _y1, x2, _y2 in windows))
        self.assertTrue(any(y2 == 2100 for _x1, _y1, _x2, y2 in windows))

    def test_schema_accepts_empty_image(self) -> None:
        document = {
            "status": "success",
            "images": [
                {
                    "image_id": "0001",
                    "file_name": "0001.jpg",
                    "width": 100,
                    "height": 80,
                    "run_end_timestamp": 1,
                    "objects": [],
                }
            ],
        }
        validate_result_document(document, ["0001.jpg"])

    def test_threshold_then_nms_keeps_best_box(self) -> None:
        detections = [
            Detection(0, 0.20, (0.0, 0.0, 10.0, 10.0)),
            Detection(0, 0.90, (0.0, 0.0, 10.0, 10.0)),
            Detection(1, 0.80, (0.0, 0.0, 10.0, 10.0)),
        ]
        filtered = apply_class_thresholds(detections, {0: 0.30, 1: 0.30})
        kept = class_aware_nms(filtered, 0.50, 100)
        self.assertEqual([(item.category_id, item.score) for item in kept], [(0, 0.90), (1, 0.80)])


class WorkspaceRegistryTests(unittest.TestCase):
    def test_frozen_registry_is_complete(self) -> None:
        root = Path(__file__).resolve().parents[1]
        train = read_split_ids(root / "data_registry/splits/v1_scene_80_20/train.txt")
        val = read_split_ids(root / "data_registry/splits/v1_scene_80_20/val.txt")
        names = load_class_names(root / "data_registry/contracts/class_names.txt")
        d0_labels = list((root / "data_registry/protocols/d0_original/labels/val").glob("*.txt"))
        self.assertEqual((len(train), len(val), len(names), len(d0_labels)), (3584, 897, 25, 897))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(set(train) | set(val)), 4481)

    def test_d0_mapping_manifest_matches_copied_labels(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "data_registry/protocols/d0_original/mapping_manifest.csv"
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 897)
        for row in rows:
            label = root / "data_registry/protocols/d0_original/labels/val" / f"{row['new_stem']}.txt"
            self.assertEqual(sha256_file(label), row["mapped_label_sha256"])


if __name__ == "__main__":
    unittest.main()
