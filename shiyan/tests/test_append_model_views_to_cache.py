import unittest

from shiyan.scripts.append_model_views_to_cache import parse_views, unflip_detections
from shiyan.src.inference.types import Detection


class AppendModelViewsTests(unittest.TestCase):
    def test_parse_views(self) -> None:
        self.assertEqual(
            parse_views(["a1280=1280", "aflip=1024,flip"]),
            {
                "a1280": {"imgsz": 1280, "horizontal_flip": False},
                "aflip": {"imgsz": 1024, "horizontal_flip": True},
            },
        )

    def test_unflip_detections(self) -> None:
        restored = unflip_detections(
            [Detection(category_id=3, score=0.75, bbox=(10.0, 20.0, 30.0, 40.0))],
            width=100,
        )
        self.assertEqual(restored[0].bbox, (70.0, 20.0, 90.0, 40.0))


if __name__ == "__main__":
    unittest.main()
