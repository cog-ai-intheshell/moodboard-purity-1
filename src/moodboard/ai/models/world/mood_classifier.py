"""Local world mood classifier adapter.

The classifier is built offline from normalized moodboard samples. At runtime
the app only loads centroid artifacts and returns nearest known mood classes or
nearest indexed world samples.
"""

from __future__ import annotations

from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.moodboard.ai.models.world.fusion_encoder import mean_dense_vector, normalize_dense_vector
from src.moodboard.ai.registry import ANALYSIS_MODEL_VERSION, SIGLIP_MODEL_ID
from src.moodboard.ai.variables.labels import clean_concept_label
from src.moodboard.analysis.scoring import cosine_similarity
from src.moodboard.core.paths import BASE_DIR, WORLD_MOOD_CLASSIFIER_PATH, WORLD_SAMPLE_INDEX_PATH
from src.moodboard.core.schemas import clamp_float


WORLD_SAMPLE_SCHEMA_VERSION = "aesthetic-world-sample-v1"


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception as exc:
        print(f"[WARN] Cannot read JSON artifact {path}: {exc}", file=sys.stderr)
    return None


def mean_pairwise_similarity(vectors: list[list[float]]) -> float:
    """Average all pairwise cosine similarities inside one class."""

    valid = [vector for vector in vectors if vector]
    if len(valid) < 2:
        return 1.0 if valid else 0.0
    values = [
        cosine_similarity(valid[i], valid[j])
        for i in range(len(valid))
        for j in range(i + 1, len(valid))
    ]
    return sum(values) / len(values) if values else 0.0


def summarize_world_class_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a classifier class with palette and modality counts."""

    palette_weights: dict[str, dict[str, Any]] = {}
    modality_counts: dict[str, Counter[str]] = {
        "colors": Counter(),
        "objects": Counter(),
        "symbols": Counter(),
        "textures": Counter(),
        "style_tags": Counter(),
        "emotion_tags": Counter(),
        "affect_tags": Counter(),
        "composition": Counter(),
    }
    for item in items:
        for color in item.get("palette", []) or []:
            if not isinstance(color, dict):
                continue
            name = str(color.get("name") or color.get("hex") or "").strip()
            if not name:
                continue
            key = f"{name}|{color.get('hex', '')}"
            if key not in palette_weights:
                palette_weights[key] = {
                    "name": name,
                    "hex": color.get("hex"),
                    "rgb": color.get("rgb"),
                    "weight": 0.0,
                }
            palette_weights[key]["weight"] += float(color.get("weight", 0.0) or 0.0)
        modalities = item.get("modalities", {}) if isinstance(item.get("modalities"), dict) else {}
        for key, counter in modality_counts.items():
            values = modalities.get(key, [])
            if not isinstance(values, list):
                continue
            counter.update(clean_concept_label(str(value), max_words=5).title() for value in values if str(value).strip())

    palette = sorted(palette_weights.values(), key=lambda value: value["weight"], reverse=True)
    total = sum(float(color.get("weight", 0.0)) for color in palette) or 1.0
    for color in palette:
        color["weight"] = round(float(color.get("weight", 0.0)) / total, 4)
    concepts = {
        key: [{"label": label, "count": count} for label, count in counter.most_common(12) if label]
        for key, counter in modality_counts.items()
    }
    return {"palette": palette[:12], "concepts": concepts}


def build_world_mood_classifier(sample_index_path: Path = WORLD_SAMPLE_INDEX_PATH) -> dict[str, Any]:
    """Build centroid classes from a previously generated world sample index."""

    payload = _read_json_file(sample_index_path)
    if not payload or payload.get("embeddingModel") != SIGLIP_MODEL_ID:
        return {
            "version": "world-mood-classifier-v1",
            "analysisModelVersion": ANALYSIS_MODEL_VERSION,
            "embeddingModel": SIGLIP_MODEL_ID,
            "updated": time.time(),
            "source": str(sample_index_path),
            "itemCount": 0,
            "items": [],
            "metrics": {"available": False, "reason": "world sample index missing or incompatible"},
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("items", []) or []:
        embedding = item.get("embedding", [])
        if not isinstance(embedding, list) or not embedding:
            continue
        label = str(item.get("dataset_origin") or "unknown").strip() or "unknown"
        grouped.setdefault(label, []).append(item)

    classes: list[dict[str, Any]] = []
    centroids: dict[str, list[float]] = {}
    for label, items in sorted(grouped.items()):
        vectors = [item["embedding"] for item in items if isinstance(item.get("embedding"), list)]
        centroid = normalize_dense_vector(mean_dense_vector(vectors) or [])
        centroids[label] = centroid
        summary = summarize_world_class_items(items)
        classes.append(
            {
                "label": label,
                "sampleCount": len(items),
                "centroid": [round(float(value), 7) for value in centroid],
                "cohesion": round(mean_pairwise_similarity(vectors), 5),
                "palette": summary["palette"],
                "concepts": summary["concepts"],
                "sampleIds": [item.get("sample_id") for item in items[:24]],
            }
        )

    inter_pairs: list[dict[str, Any]] = []
    labels = [item["label"] for item in classes]
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            similarity = cosine_similarity(centroids[left], centroids[right])
            inter_pairs.append({"left": left, "right": right, "similarity": round(similarity, 5)})
    mean_intra = sum(float(item.get("cohesion", 0.0)) for item in classes) / len(classes) if classes else 0.0
    mean_inter = sum(pair["similarity"] for pair in inter_pairs) / len(inter_pairs) if inter_pairs else 0.0
    separation = clamp_float((mean_intra - mean_inter + 1.0) / 2.0, 0.0, 0.0, 1.0)

    return {
        "version": "world-mood-classifier-v1",
        "analysisModelVersion": ANALYSIS_MODEL_VERSION,
        "schemaVersion": WORLD_SAMPLE_SCHEMA_VERSION,
        "embeddingModel": SIGLIP_MODEL_ID,
        "source": str(sample_index_path.relative_to(BASE_DIR)) if sample_index_path.is_relative_to(BASE_DIR) else str(sample_index_path),
        "updated": time.time(),
        "itemCount": len(classes),
        "sampleCount": sum(len(items) for items in grouped.values()),
        "method": "supervised centroid classifier over normalized moodboard sample embeddings",
        "items": classes,
        "metrics": {
            "available": bool(classes),
            "classCount": len(classes),
            "meanIntraSimilarity": round(mean_intra, 5),
            "meanInterSimilarity": round(mean_inter, 5),
            "separation": round(separation, 5),
            "interClassSimilarities": inter_pairs,
        },
    }


def nearest_world_moods(vector: list[float] | None, limit: int = 5) -> list[dict[str, Any]]:
    """Return nearest trained mood classes for a fused moodboard vector."""

    if not vector:
        return []
    payload = _read_json_file(WORLD_MOOD_CLASSIFIER_PATH)
    if not payload or payload.get("embeddingModel") != SIGLIP_MODEL_ID:
        return []
    normalized = normalize_dense_vector(vector)
    ranked: list[dict[str, Any]] = []
    for item in payload.get("items", []) or []:
        centroid = item.get("centroid", [])
        if not isinstance(centroid, list) or len(centroid) != len(normalized):
            continue
        similarity = cosine_similarity(normalized, centroid)
        ranked.append(
            {
                "label": item.get("label"),
                "similarity": round(similarity, 5),
                "confidence": round(clamp_float((similarity + 1.0) / 2.0, 0.0, 0.0, 1.0), 4),
                "sampleCount": item.get("sampleCount"),
                "cohesion": item.get("cohesion"),
                "palette": item.get("palette", [])[:8],
                "concepts": item.get("concepts", {}),
            }
        )
    ranked.sort(key=lambda value: value["similarity"], reverse=True)
    return ranked[:limit]


def nearest_world_samples(vector: list[float] | None, limit: int = 8) -> list[dict[str, Any]]:
    """Return nearest indexed world samples for a fused moodboard vector."""

    if not vector:
        return []
    payload = _read_json_file(WORLD_SAMPLE_INDEX_PATH)
    if not payload or payload.get("embeddingModel") != SIGLIP_MODEL_ID:
        return []
    ranked = []
    for item in payload.get("items", []) or []:
        embedding = item.get("embedding", [])
        if not isinstance(embedding, list) or len(embedding) != len(vector):
            continue
        ranked.append(
            {
                "sampleId": item.get("sample_id"),
                "datasetOrigin": item.get("dataset_origin"),
                "assetRef": item.get("asset_ref"),
                "similarity": round(cosine_similarity(vector, embedding), 5),
                "palette": item.get("palette", [])[:6],
                "modalities": item.get("modalities", {}),
            }
        )
    ranked.sort(key=lambda value: value["similarity"], reverse=True)
    return ranked[:limit]


def nearest(vector: list[float] | None, limit: int = 5) -> list[dict[str, Any]]:
    """Registry-friendly alias for class retrieval."""

    return nearest_world_moods(vector, limit)
