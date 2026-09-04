"""Pure-Python post-processing for dual-model competition experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .labels import class_group


GROUPS = ("ship", "aircraft", "vehicle")
DUAL_SOURCES = ("a", "b")


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    left_box = tuple(float(value) for value in left)
    right_box = tuple(float(value) for value in right)
    if len(left_box) != 4 or len(right_box) != 4:
        raise ValueError("boxes must contain four xyxy coordinates")
    x1 = max(left_box[0], right_box[0])
    y1 = max(left_box[1], right_box[1])
    x2 = min(left_box[2], right_box[2])
    y2 = min(left_box[3], right_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left_box[2] - left_box[0]) * max(0.0, left_box[3] - left_box[1])
    right_area = max(0.0, right_box[2] - right_box[0]) * max(0.0, right_box[3] - right_box[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _normalized_detection(item: Mapping[str, Any], source: str) -> dict[str, Any]:
    source = str(source).strip()
    if not source:
        raise ValueError("source must be a non-empty string")
    category_id = int(item["category_id"])
    score = float(item["score"])
    bbox = [float(value) for value in item["bbox"]]
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    return {
        "category_id": category_id,
        "score": score,
        "bbox": bbox,
        "group": class_group(category_id),
        "sources": (source,),
        "support": 1,
    }


def _preferred_category(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    source_preference: Mapping[str, str],
) -> int:
    left_category = int(left["category_id"])
    right_category = int(right["category_id"])
    if left_category == right_category:
        return left_category
    group = str(left["group"])
    preferred_source = source_preference.get(group, "max")
    if preferred_source == "a":
        return left_category
    if preferred_source == "b":
        return right_category
    if preferred_source != "max":
        raise ValueError(f"invalid source preference for {group}: {preferred_source}")
    return left_category if float(left["score"]) >= float(right["score"]) else right_category


def _fuse_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    source_preference: Mapping[str, str],
) -> dict[str, Any]:
    left_score = float(left["score"])
    right_score = float(right["score"])
    weight_sum = max(left_score + right_score, 1e-12)
    bbox = [
        (left_score * float(left["bbox"][index]) + right_score * float(right["bbox"][index])) / weight_sum
        for index in range(4)
    ]
    return {
        "category_id": _preferred_category(left, right, source_preference),
        "score": max(left_score, right_score),
        "bbox": bbox,
        "group": str(left["group"]),
        "sources": ("a", "b"),
        "support": 2,
    }


def match_dual_model_detections(
    model_a: Iterable[Mapping[str, Any]],
    model_b: Iterable[Mapping[str, Any]],
    *,
    consensus_iou: float,
    source_preference: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedily match same-group boxes and return fused, A-only and B-only detections."""

    if not 0.0 < consensus_iou <= 1.0:
        raise ValueError("consensus_iou must be in (0, 1]")
    left_items = [_normalized_detection(item, "a") for item in model_a]
    right_items = [_normalized_detection(item, "b") for item in model_b]
    pairs: list[tuple[int, float, float, int, int]] = []
    for left_index, left in enumerate(left_items):
        for right_index, right in enumerate(right_items):
            if left["group"] != right["group"]:
                continue
            overlap = box_iou(left["bbox"], right["bbox"])
            if overlap < consensus_iou:
                continue
            same_class = int(left["category_id"]) == int(right["category_id"])
            score_sum = float(left["score"]) + float(right["score"])
            pairs.append((1 if same_class else 0, overlap, score_sum, left_index, right_index))
    pairs.sort(reverse=True)

    matched_left: set[int] = set()
    matched_right: set[int] = set()
    fused: list[dict[str, Any]] = []
    for _, _, _, left_index, right_index in pairs:
        if left_index in matched_left or right_index in matched_right:
            continue
        matched_left.add(left_index)
        matched_right.add(right_index)
        fused.append(_fuse_pair(left_items[left_index], right_items[right_index], source_preference))
    left_only = [item for index, item in enumerate(left_items) if index not in matched_left]
    right_only = [item for index, item in enumerate(right_items) if index not in matched_right]
    return fused, left_only, right_only


def combine_models(
    model_a: Iterable[Mapping[str, Any]],
    model_b: Iterable[Mapping[str, Any]],
    *,
    fusion_mode: str,
    consensus_iou: float,
    source_preference: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Combine two frozen-model outputs using a declared inference-time policy."""

    if fusion_mode == "source-a":
        return [_normalized_detection(item, "a") for item in model_a]
    if fusion_mode == "source-b":
        return [_normalized_detection(item, "b") for item in model_b]
    if fusion_mode == "route":
        left = [_normalized_detection(item, "a") for item in model_a]
        right = [_normalized_detection(item, "b") for item in model_b]
        routed: list[dict[str, Any]] = []
        for group in GROUPS:
            selected_source = source_preference.get(group, "a")
            if selected_source not in DUAL_SOURCES:
                raise ValueError(f"route mode requires source a or b for {group}")
            selected = left if selected_source == "a" else right
            routed.extend(item for item in selected if item["group"] == group)
        return routed

    if fusion_mode == "union":
        return [
            *[_normalized_detection(item, "a") for item in model_a],
            *[_normalized_detection(item, "b") for item in model_b],
        ]
    if fusion_mode not in {"consensus", "intersection"}:
        raise ValueError(f"unknown fusion_mode: {fusion_mode}")
    fused, left_only, right_only = match_dual_model_detections(
        model_a,
        model_b,
        consensus_iou=consensus_iou,
        source_preference=source_preference,
    )
    return fused if fusion_mode == "intersection" else fused + left_only + right_only


def _fuse_cluster(
    members: list[dict[str, Any]],
    *,
    preferred_source: str,
    score_aggregation: str,
) -> dict[str, Any]:
    if not members:
        raise ValueError("cannot fuse an empty cluster")
    scores = [float(item["score"]) for item in members]
    weight_sum = max(sum(scores), 1e-12)
    bbox = [
        sum(score * float(item["bbox"][index]) for score, item in zip(scores, members))
        / weight_sum
        for index in range(4)
    ]
    preferred = next(
        (item for item in members if preferred_source in item["sources"]),
        None,
    )
    if preferred is not None:
        category_id = int(preferred["category_id"])
    else:
        category_scores: dict[int, float] = {}
        category_max: dict[int, float] = {}
        for item in members:
            category_id = int(item["category_id"])
            score = float(item["score"])
            category_scores[category_id] = category_scores.get(category_id, 0.0) + score
            category_max[category_id] = max(category_max.get(category_id, 0.0), score)
        category_id = max(
            category_scores,
            key=lambda item: (category_scores[item], category_max[item], -item),
        )
    if score_aggregation == "max":
        score = max(scores)
    elif score_aggregation == "mean":
        score = sum(scores) / len(scores)
    else:
        raise ValueError(f"unknown score aggregation: {score_aggregation}")
    sources = tuple(source for item in members for source in item["sources"])
    return {
        "category_id": category_id,
        "score": score,
        "bbox": bbox,
        "group": str(members[0]["group"]),
        "sources": sources,
        "support": len(set(sources)),
    }


def cluster_multiple_model_detections(
    model_outputs: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    consensus_iou: float,
    source_preference: Mapping[str, str],
    score_aggregation: str = "max",
) -> list[dict[str, Any]]:
    """Cluster same-group detections while allowing at most one vote per model."""

    if not 0.0 < consensus_iou <= 1.0:
        raise ValueError("consensus_iou must be in (0, 1]")
    if not model_outputs:
        raise ValueError("model_outputs must not be empty")
    sources = tuple(model_outputs)
    for group in GROUPS:
        preferred = source_preference.get(group, "max")
        if preferred != "max" and preferred not in sources:
            raise ValueError(f"unknown preferred source for {group}: {preferred}")

    candidates = [
        _normalized_detection(item, source)
        for source, detections in model_outputs.items()
        for item in detections
    ]
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    clusters: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        source = str(candidate["sources"][0])
        best_index: int | None = None
        best_rank = (-1, -1.0)
        for cluster_index, cluster in enumerate(clusters):
            if str(cluster[0]["group"]) != str(candidate["group"]):
                continue
            if any(source in member["sources"] for member in cluster):
                continue
            current = _fuse_cluster(
                cluster,
                preferred_source=source_preference.get(str(candidate["group"]), "max"),
                score_aggregation=score_aggregation,
            )
            overlap = box_iou(candidate["bbox"], current["bbox"])
            if overlap < consensus_iou:
                continue
            same_class = any(
                int(member["category_id"]) == int(candidate["category_id"])
                for member in cluster
            )
            rank = (1 if same_class else 0, overlap)
            if rank > best_rank:
                best_index = cluster_index
                best_rank = rank
        if best_index is None:
            clusters.append([candidate])
        else:
            clusters[best_index].append(candidate)

    return [
        _fuse_cluster(
            cluster,
            preferred_source=source_preference.get(str(cluster[0]["group"]), "max"),
            score_aggregation=score_aggregation,
        )
        for cluster in clusters
    ]


def combine_multiple_models(
    model_outputs: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    fusion_mode: str,
    consensus_iou: float,
    source_preference: Mapping[str, str],
    minimum_support: int = 2,
    score_aggregation: str = "max",
) -> list[dict[str, Any]]:
    """Combine any number of frozen model outputs under one declared policy."""

    if not model_outputs:
        raise ValueError("model_outputs must not be empty")
    sources = tuple(model_outputs)
    if minimum_support <= 0 or minimum_support > len(sources):
        raise ValueError("minimum_support must be between 1 and the model count")
    if fusion_mode == "route":
        routed: list[dict[str, Any]] = []
        for group in GROUPS:
            selected_source = source_preference.get(group, "")
            if selected_source not in model_outputs:
                raise ValueError(f"route mode has no valid source for {group}: {selected_source}")
            routed.extend(
                item
                for item in (
                    _normalized_detection(raw, selected_source)
                    for raw in model_outputs[selected_source]
                )
                if item["group"] == group
            )
        return routed
    if fusion_mode == "union":
        return [
            _normalized_detection(item, source)
            for source, detections in model_outputs.items()
            for item in detections
        ]
    if fusion_mode not in {"consensus", "vote"}:
        raise ValueError(f"unknown multi-model fusion mode: {fusion_mode}")
    clustered = cluster_multiple_model_detections(
        model_outputs,
        consensus_iou=consensus_iou,
        source_preference=source_preference,
        score_aggregation=score_aggregation,
    )
    if fusion_mode == "vote":
        return [item for item in clustered if int(item["support"]) >= minimum_support]
    return clustered


def group_aware_nms(
    detections: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float,
    max_detections: int = 300,
) -> list[dict[str, Any]]:
    """Suppress overlapping detections across fine classes within each competition group."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    ordered = sorted(
        (dict(item) for item in detections),
        key=lambda item: (float(item["score"]), int(item.get("support", 1))),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        suppressed = any(
            str(candidate["group"]) == str(previous["group"])
            and box_iou(candidate["bbox"], previous["bbox"]) >= iou_threshold
            for previous in kept
        )
        if not suppressed:
            kept.append(candidate)
        if len(kept) >= max_detections:
            break
    return kept


def apply_modality_and_thresholds(
    detections: Iterable[Mapping[str, Any]],
    *,
    modality: str,
    modality_policy: str,
    consensus_thresholds: Mapping[str, float],
    single_thresholds: Mapping[str, float],
    ship_color_conf: float,
    nonship_gray_conf: float,
    class_thresholds: Mapping[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply support-aware thresholds and optional bidirectional sensor routing."""

    if modality not in {"grayscale", "color", "uncertain"}:
        raise ValueError(f"unknown modality: {modality}")
    if modality_policy not in {"off", "soft", "strict"}:
        raise ValueError(f"unknown modality_policy: {modality_policy}")
    per_class = {int(key): float(value) for key, value in (class_thresholds or {}).items()}
    kept: list[dict[str, Any]] = []
    for detection in detections:
        item = dict(detection)
        group = str(item["group"])
        support = int(item.get("support", 1))
        thresholds = consensus_thresholds if support >= 2 else single_thresholds
        threshold = float(thresholds[group])
        threshold = max(threshold, per_class.get(int(item["category_id"]), 0.0))
        cross_modality = (modality == "color" and group == "ship") or (
            modality == "grayscale" and group in {"aircraft", "vehicle"}
        )
        if cross_modality and modality_policy == "strict":
            continue
        if cross_modality and modality_policy == "soft":
            threshold = max(
                threshold,
                ship_color_conf if group == "ship" else nonship_gray_conf,
            )
        if float(item["score"]) >= threshold:
            kept.append(item)
    return kept


def official_object(item: Mapping[str, Any], class_names: tuple[str, ...]) -> dict[str, Any]:
    category_id = int(item["category_id"])
    return {
        "category_id": category_id,
        "category_name": class_names[category_id],
        "score": round(float(item["score"]), 6),
        "bbox": [round(float(value), 4) for value in item["bbox"]],
    }
