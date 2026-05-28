"""Moodboard graph construction.

The graph is the bridge between image-level embeddings and the UI. Nodes are
images or extracted modalities; edges represent similarity, co-occurrence or
affinity. Keeping this here lets API/server code stay unaware of graph details.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from src.moodboard.analysis.clustering import vector_projection_2d
from src.moodboard.analysis.scoring import cosine_similarity


CLUSTER_COLORS = ["#5D71FC", "#EB5757", "#f89540", "#27AE60", "#A855F7", "#F2C94C", "#56CCF2", "#FF6FB1"]


def clamp_float(value: float, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(low, min(high, number))


def cluster_color(cluster: int) -> str:
    """Return the stable design-system color for a cluster id."""

    return CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)]


def canonical_color_key(name: str) -> str:
    """Collapse color labels so shared colors create one graph node."""

    key = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return key or "color"


def weighted_centroid(vectors: list[list[float]], weighted_indices: list[tuple[int, float]]) -> list[float]:
    """Average vectors using edge/modal association strengths as weights."""

    if not vectors or not weighted_indices:
        return []
    valid = [(idx, max(0.0, float(weight))) for idx, weight in weighted_indices if 0 <= idx < len(vectors)]
    if not valid:
        return []
    size = min(len(vectors[idx]) for idx, _weight in valid)
    if size <= 0:
        return []
    total = sum(weight for _idx, weight in valid)
    if total <= 1e-9:
        total = float(len(valid))
        valid = [(idx, 1.0) for idx, _weight in valid]
    centroid = [0.0] * size
    for idx, amount in valid:
        for dim in range(size):
            centroid[dim] += vectors[idx][dim] * amount
    return [value / total for value in centroid]


def dominant_cluster_for_indices(image_entries: list[dict[str, Any]], weighted_indices: list[tuple[int, float]]) -> int:
    """Assign modality nodes to the image cluster that contributes most weight."""

    cluster_weights: Counter[int] = Counter()
    for idx, weight in weighted_indices:
        if 0 <= idx < len(image_entries):
            cluster_weights[int(image_entries[idx].get("cluster", 0))] += max(0.0, float(weight))
    if not cluster_weights:
        return 0
    return int(cluster_weights.most_common(1)[0][0])


def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_analysis_graph(*args, **kwargs)


def build_analysis_graph(
    image_entries: list[dict[str, Any]],
    vectors: list[list[float]],
    global_palette: list[dict[str, Any]],
    emotion_scores: dict[str, float],
    aesthetic_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    node_vectors: dict[str, list[float]] = {}

    def add_node(
        node_id: str,
        node_type: str,
        label: str,
        cluster: int,
        weight: float,
        vector: list[float],
        associated_indices: list[int],
    ) -> None:
        if node_id in node_ids:
            return
        node_ids.add(node_id)
        cluster_id = int(cluster)
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "cluster": cluster_id,
                "clusterLabel": f"Cluster {cluster_id + 1}",
                "clusterColor": cluster_color(cluster_id),
                "weight": round(weight, 4),
                "x": 0.0,
                "y": 0.0,
                "associatedImages": [image_entries[idx]["id"] for idx in sorted(set(associated_indices)) if 0 <= idx < len(image_entries)],
            }
        )
        node_vectors[node_id] = vector

    def add_edge(source: str, target: str, edge_type: str, weight: float) -> None:
        if source == target or source not in node_ids or target not in node_ids:
            return
        if weight <= 0.0:
            return
        edges.append({"source": source, "target": target, "type": edge_type, "weight": round(clamp_float(weight, 0.0, 0.0, 1.0), 4)})

    def add_attribute_node(
        node_id: str,
        node_type: str,
        label: str,
        weighted_indices: list[tuple[int, float]],
        weight: float,
    ) -> bool:
        weighted_indices = [(idx, max(0.0, float(amount))) for idx, amount in weighted_indices if 0 <= idx < len(image_entries) and amount > 0]
        if not weighted_indices:
            return False
        cluster_id = dominant_cluster_for_indices(image_entries, weighted_indices)
        vector = weighted_centroid(vectors, weighted_indices)
        if not vector:
            return False
        add_node(node_id, node_type, label, cluster_id, weight, vector, [idx for idx, _amount in weighted_indices])
        return True

    for idx, entry in enumerate(image_entries):
        add_node(entry["id"], "image", entry["filename"], int(entry["cluster"]), 0.72 + entry["scores"]["hero"] * 0.25, vectors[idx], [idx])

    for idx, entry in enumerate(image_entries):
        ranked_visual: list[tuple[int, float]] = []
        ranked_attention: list[tuple[int, float]] = []
        attention_vector = (entry.get("attention") or {}).get("embedding", [])
        for other_idx, other in enumerate(image_entries):
            if idx >= other_idx:
                continue
            visual_similarity = cosine_similarity(vectors[idx], vectors[other_idx])
            ranked_visual.append((other_idx, clamp_float((visual_similarity + 1.0) / 2.0, 0.0, 0.0, 1.0)))
            other_attention = (other.get("attention") or {}).get("embedding", [])
            if isinstance(attention_vector, list) and isinstance(other_attention, list) and attention_vector and other_attention:
                size = min(len(attention_vector), len(other_attention))
                if size > 1:
                    uniform = 1.0 / size
                    left = [float(value) - uniform for value in attention_vector[:size]]
                    right = [float(value) - uniform for value in other_attention[:size]]
                    attention_similarity = cosine_similarity(left, right)
                    ranked_attention.append((other_idx, clamp_float((attention_similarity + 1.0) / 2.0, 0.0, 0.0, 1.0)))
        for other_idx, weight in sorted(ranked_visual, key=lambda item: item[1], reverse=True)[:4]:
            if weight >= 0.54:
                add_edge(entry["id"], image_entries[other_idx]["id"], "image_similarity", weight)
        for other_idx, weight in sorted(ranked_attention, key=lambda item: item[1], reverse=True)[:2]:
            if weight >= 0.55:
                add_edge(entry["id"], image_entries[other_idx]["id"], "attention_similarity", weight)

    for item in global_palette:
        color_key = canonical_color_key(str(item.get("name", "")))
        node_id = "color-" + color_key
        weighted_indices = []
        for idx, entry in enumerate(image_entries):
            local_colors = [
                color
                for color in entry.get("palette", [])
                if canonical_color_key(str(color.get("name", ""))) == color_key
            ]
            if local_colors:
                local_weight = sum(float(color.get("weight", item.get("weight", 0.2)) or 0.0) for color in local_colors)
                weighted_indices.append((idx, local_weight))
        if add_attribute_node(node_id, "color", item["name"], weighted_indices, max(0.035, float(item.get("weight", 0.05)))):
            for idx, amount in weighted_indices:
                add_edge(image_entries[idx]["id"], node_id, "color_affinity", amount)

    top_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)[:6]
    for emotion, score in top_emotions:
        node_id = f"emotion-{emotion}"
        weighted_indices = [(idx, float(entry.get("emotionScores", {}).get(emotion, 0.0))) for idx, entry in enumerate(image_entries)]
        if add_attribute_node(node_id, "emotion", emotion.title(), weighted_indices, score):
            for idx, amount in weighted_indices:
                if amount > 0:
                    add_edge(image_entries[idx]["id"], node_id, "emotion_affinity", amount)

    def term_confidence(entry: dict[str, Any], node_type: str, term: str) -> float:
        expected = {
            "symbol": {"symbol", "tag"},
            "object": {"object"},
            "texture": {"texture"},
            "style": {"style"},
            "affect": {"affect"},
            "composition": {"composition"},
        }.get(node_type, {node_type})
        matches = [
            float(observation.get("confidence", 0.55))
            for observation in entry.get("observations", [])
            if str(observation.get("type")) in expected and str(observation.get("label", "")).lower() == term.lower()
        ]
        return max(matches) if matches else 0.55

    term_types = [
        ("objects", "object"),
        ("symbols", "symbol"),
        ("tags", "symbol"),
        ("textures", "texture"),
        ("styles", "style"),
        ("affects", "affect"),
        ("composition", "composition"),
    ]
    seen_term_nodes: set[str] = set()
    for key, node_type in term_types:
        counts: Counter[str] = Counter()
        for entry in image_entries:
            counts.update(entry.get(key, []))
        limit = 18 if node_type in {"symbol", "object", "affect"} else (14 if node_type in {"style", "texture"} else 8)
        for term, amount in counts.most_common(limit):
            node_id = f"{node_type}-" + re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
            if node_id in seen_term_nodes:
                continue
            seen_term_nodes.add(node_id)
            weighted_indices = [(idx, term_confidence(entry, node_type, term)) for idx, entry in enumerate(image_entries) if term in entry.get(key, [])]
            if add_attribute_node(node_id, node_type, term.title(), weighted_indices, amount / max(1, len(image_entries))):
                for idx, confidence in weighted_indices:
                    edge_type = "co_occurrence" if node_type in {"symbol", "object", "affect"} else f"{node_type}_affinity"
                    add_edge(image_entries[idx]["id"], node_id, edge_type, max(0.28, confidence))

    for idx, match in enumerate(aesthetic_matches[:5]):
        node_id = "aesthetic-" + re.sub(r"[^a-z0-9]+", "-", match["name"].lower()).strip("-")
        keywords = set(str(value).lower() for value in match.get("keywords", []))
        weighted_indices = []
        for image_idx, entry in enumerate(image_entries):
            terms = set(
                str(value).lower()
                for value in (
                    entry.get("tags", [])
                    + entry.get("objects", [])
                    + entry.get("symbols", [])
                    + entry.get("textures", [])
                    + entry.get("styles", [])
                    + entry.get("composition", [])
                )
            )
            terms.update(str(color.get("name", "")).lower() for color in entry.get("palette", []))
            terms.update(str(emotion).lower() for emotion in entry.get("emotionScores", {}).keys())
            overlap = len(terms & keywords)
            relevance = float(match.get("score", 0.0)) * (0.22 + min(1.0, overlap / 4.0) * 0.78)
            if overlap or idx == 0:
                weighted_indices.append((image_idx, relevance))
        if add_attribute_node(node_id, "aesthetic", match["name"], weighted_indices, match["score"]):
            for image_idx, relevance in weighted_indices:
                if relevance > 0.08:
                    add_edge(image_entries[image_idx]["id"], node_id, "aesthetic_match", relevance)

    projection_vectors = [node_vectors[node["id"]] for node in nodes]
    coords, projection_method = vector_projection_2d(projection_vectors)
    for node, (x, y) in zip(nodes, coords):
        node["x"] = x
        node["y"] = y
    edges.sort(key=lambda edge: edge["weight"], reverse=True)
    return {
        "nodes": nodes,
        "edges": edges[:1200],
        "projection": {
            "method": projection_method,
            "basis": "node vectors: image embeddings plus patch-attention similarity; attribute vectors are weighted centroids of associated image embeddings",
            "distance": "cosine",
        },
        "clusterColors": {str(idx): cluster_color(idx) for idx in sorted({int(node["cluster"]) for node in nodes})},
    }
