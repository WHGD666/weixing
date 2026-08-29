"""Post-processing filters that preserve the official result schema."""

from collections.abc import Mapping

from .schema import validate_result_document


def filter_class_thresholds(
    document: Mapping[str, object],
    thresholds: Mapping[int, float],
) -> dict[str, object]:
    """Filter only selected classes by score; unspecified classes are unchanged."""

    validate_result_document(document)
    normalized = {int(category_id): float(value) for category_id, value in thresholds.items()}
    for category_id, threshold in normalized.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]: {category_id}={threshold}")

    filtered_images: list[dict[str, object]] = []
    for image in document["images"]:
        filtered_image = dict(image)
        filtered_image["objects"] = [
            dict(item)
            for item in image["objects"]
            if float(item["score"]) >= normalized.get(int(item["category_id"]), 0.0)
        ]
        filtered_images.append(filtered_image)
    filtered = {"status": document["status"], "images": filtered_images}
    validate_result_document(filtered)
    return filtered
