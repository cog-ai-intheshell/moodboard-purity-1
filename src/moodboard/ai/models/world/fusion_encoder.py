"""World-model fusion adapter.

The fusion encoder combines visual SigLIP vectors with per-modality text
evidence. It is still a deterministic MVP encoder, but it has the same boundary
as the future learned fusion model: image entries in, unified aesthetic vectors
out.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from src.moodboard.ai.models.embeddings.siglip2 import try_siglip_text_embeddings
from src.moodboard.ai.variables.labels import OBSERVATION_FIELD_BY_TYPE, clean_concept_label
from src.moodboard.core.cache import ML_MODEL_CACHE
from src.moodboard.core.paths import FUSION_CALIBRATOR_PATH
from src.moodboard.core.schemas import clamp_float


MODALITY_FUSION_WEIGHTS = {
    "visual": 0.50,
    "color": 0.12,
    "object": 0.10,
    "symbol": 0.10,
    "style": 0.08,
    "emotion": 0.05,
    "affect": 0.05,
    "composition": 0.04,
    "texture": 0.04,
    "tag": 0.03,
}


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception as exc:
        print(f"[WARN] Cannot read JSON artifact {path}: {exc}", file=sys.stderr)
    return None


def load_fusion_weights() -> dict[str, float]:
    """Load calibrated modality weights, falling back to sane MVP defaults."""

    if "fusion_weights" in ML_MODEL_CACHE:
        return dict(ML_MODEL_CACHE["fusion_weights"])
    weights = dict(MODALITY_FUSION_WEIGHTS)
    payload = _read_json_file(FUSION_CALIBRATOR_PATH)
    if payload:
        raw_weights = payload.get("weights", {})
        if isinstance(raw_weights, dict):
            for key, value in raw_weights.items():
                if key in weights:
                    weights[key] = clamp_float(float(value), weights[key], 0.0, 1.0)
    total = sum(max(0.0, value) for value in weights.values())
    if total > 1e-9:
        weights = {key: value / total for key, value in weights.items()}
    ML_MODEL_CACHE["fusion_weights"] = weights
    return dict(weights)


def normalize_dense_vector(vector: list[float]) -> list[float]:
    """Return an L2-normalized dense vector without changing dimensionality."""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        return list(vector)
    return [value / norm for value in vector]


def mean_dense_vector(vectors: list[list[float]]) -> list[float] | None:
    """Average a list of dense vectors, guarding against empty inputs."""

    valid = [vector for vector in vectors if vector]
    if not valid:
        return None
    size = min(len(vector) for vector in valid)
    if size <= 0:
        return None
    return [sum(vector[idx] for vector in valid) / len(valid) for idx in range(size)]


def modality_phrase_for_entry(entry: dict[str, Any], obs_type: str, limit: int = 8) -> str:
    """Convert one modality of an image entry into a concise text phrase."""

    if obs_type == "color":
        labels = [str(color.get("name", "")) for color in entry.get("palette", [])[:8]]
    elif obs_type == "emotion":
        labels = [label for label, _value in sorted(entry.get("emotionScores", {}).items(), key=lambda item: item[1], reverse=True)[:limit]]
    elif obs_type == "affect":
        labels = entry.get("affects", [])[:limit]
    else:
        field = OBSERVATION_FIELD_BY_TYPE.get(obs_type)
        labels = list(entry.get(field, []) if field else [])
    cleaned = []
    for label in labels:
        clean = clean_concept_label(str(label), max_words=5)
        if clean and clean not in cleaned:
            cleaned.append(clean)
        if len(cleaned) >= limit:
            break
    if not cleaned:
        return ""
    if obs_type == "color":
        return "palette: " + ", ".join(cleaned)
    if obs_type == "emotion":
        return "affective mood: " + ", ".join(cleaned)
    if obs_type == "affect":
        return "symbolic affective values: " + ", ".join(cleaned)
    return f"{obs_type}: " + ", ".join(cleaned)


def build_composite_vectors(
    image_entries: list[dict[str, Any]],
    visual_vectors: list[list[float]],
) -> tuple[list[list[float]], str]:
    """Fuse visual vectors with text embeddings for each detected modality."""

    if not image_entries or not visual_vectors:
        return visual_vectors, "no-vectors"
    phrase_refs: list[tuple[int, str]] = []
    phrase_texts: list[str] = []
    for idx, entry in enumerate(image_entries):
        entry_summary: dict[str, list[str]] = {}
        for obs_type in ("color", "object", "symbol", "texture", "style", "emotion", "affect", "composition", "tag"):
            phrase = modality_phrase_for_entry(entry, obs_type)
            if not phrase:
                continue
            phrase_refs.append((idx, obs_type))
            phrase_texts.append(f"image modality evidence - {phrase}")
            entry_summary[obs_type] = phrase.split(":", 1)[-1].strip().split(", ")
        entry["modalities"] = entry_summary

    if not phrase_texts:
        return visual_vectors, "visual-only"
    text_vectors, text_status = try_siglip_text_embeddings(phrase_texts)
    if not text_vectors or len(text_vectors) != len(phrase_refs):
        return visual_vectors, f"visual-only; modality text unavailable ({text_status})"
    if not visual_vectors or len(text_vectors[0]) != len(visual_vectors[0]):
        return visual_vectors, "visual-only; modality dimension mismatch"

    grouped: dict[int, dict[str, list[float]]] = {}
    for (idx, obs_type), vector in zip(phrase_refs, text_vectors):
        grouped.setdefault(idx, {})[obs_type] = vector

    fusion_weights = load_fusion_weights()
    composite: list[list[float]] = []
    for idx, visual_vector in enumerate(visual_vectors):
        size = len(visual_vector)
        accumulator = [0.0] * size
        total_weight = 0.0
        pieces = {"visual": normalize_dense_vector(visual_vector), **grouped.get(idx, {})}
        for obs_type, vector in pieces.items():
            if len(vector) != size:
                continue
            weight = fusion_weights.get(obs_type, 0.02)
            total_weight += weight
            normalized = normalize_dense_vector(vector)
            for dim in range(size):
                accumulator[dim] += normalized[dim] * weight
        if total_weight <= 1e-9:
            composite.append(visual_vector)
        else:
            composite.append(normalize_dense_vector([value / total_weight for value in accumulator]))
    return composite, f"weighted-multimodal-fusion; {len(phrase_texts)} modality phrases; {text_status}"


def composite_vectors(entries: list[dict[str, Any]], visual_vectors: list[list[float]]) -> tuple[list[list[float]], str]:
    """Registry-friendly alias for multimodal fusion."""

    return build_composite_vectors(entries, visual_vectors)


def fusion_weights() -> dict[str, float]:
    """Registry-friendly alias for calibrated fusion weights."""

    return load_fusion_weights()
