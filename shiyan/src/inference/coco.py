"""Conversion from the Docker result contract to COCO detection JSON."""

from collections.abc import Mapping

from .schema import validate_result_document


def result_to_coco(document: Mapping[str, object]) -> list[dict[str, object]]:
    """Convert official xyxy pixel boxes to COCO xywh pixel boxes."""

    validate_result_document(document)
    rows: list[dict[str, object]] = []
    for image in document["images"]:
        for detection in image["objects"]:
            x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
            rows.append(
                {
                    "image_id": image["image_id"],
                    "category_id": detection["category_id"],
                    "bbox": [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)],
                    "score": round(float(detection["score"]), 6),
                }
            )
    return rows
