"""World-sample embedding index builder.

Offline learning jobs use this module to turn normalized JSONL samples into a
local nearest-neighbor artifact. The web app later loads that artifact; it does
not rebuild it during startup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.moodboard.ai.models.embeddings.siglip2 import try_siglip_embeddings, try_siglip_text_embeddings
from src.moodboard.ai.models.world.fusion_encoder import normalize_dense_vector
from src.moodboard.ai.registry import ANALYSIS_MODEL_VERSION, SIGLIP_MODEL_ID
from src.moodboard.analysis.scoring import cosine_similarity
from src.moodboard.core.image_io import VALID_EXTENSIONS
from src.moodboard.core.paths import BASE_DIR, WORLD_SAMPLES_PATH


WORLD_SAMPLE_SCHEMA_VERSION = "aesthetic-world-sample-v1"


def load_world_samples(path: Path = WORLD_SAMPLES_PATH, max_items: int | None = None) -> list[dict[str, Any]]:
    """Read normalized world-learning samples from JSONL."""

    samples: list[dict[str, Any]] = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(sample, dict):
                samples.append(sample)
            if max_items and len(samples) >= max_items:
                break
    return samples


def sample_asset_path(sample: dict[str, Any]) -> Path | None:
    """Resolve a sample image path without supporting asset redistribution."""

    asset_ref = sample.get("asset_ref") or {}
    if not isinstance(asset_ref, dict):
        return None
    if asset_ref.get("kind") != "local_path":
        return None
    value = str(asset_ref.get("value", "")).strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path if path.exists() else None


def world_sample_signature_text(sample: dict[str, Any]) -> str:
    """Describe sample metadata as compact text for multimodal fusion."""

    palette = sample.get("palette", []) if isinstance(sample.get("palette"), list) else []
    modalities = sample.get("modalities", {}) if isinstance(sample.get("modalities"), dict) else {}
    metadata = sample.get("raw_metadata", {}) if isinstance(sample.get("raw_metadata"), dict) else {}
    parts: list[str] = []

    color_terms = []
    for color in palette[:12]:
        if not isinstance(color, dict):
            continue
        name = str(color.get("name") or "").strip()
        role = str(color.get("role") or "").strip()
        if name:
            color_terms.append(f"{role} {name}".strip())
    if color_terms:
        parts.append("palette: " + ", ".join(color_terms))

    for key in ("objects", "symbols", "textures", "style_tags", "emotion_tags", "affect_tags", "composition", "colors"):
        values = modalities.get(key, [])
        if isinstance(values, list) and values:
            label = key.replace("_tags", "").replace("_", " ")
            parts.append(f"{label}: " + ", ".join(str(value) for value in values[:12] if str(value).strip()))

    orientation = str(metadata.get("orientation") or "").strip()
    if orientation:
        parts.append(f"composition: {orientation}")
    brightness = metadata.get("brightness")
    if isinstance(brightness, int | float):
        if brightness < 0.32:
            parts.append("lighting: low key, dark tonal range")
        elif brightness > 0.72:
            parts.append("lighting: high key, bright tonal range")
        else:
            parts.append("lighting: balanced tonal range")

    text = "; ".join(part for part in parts if part)
    return ("aesthetic world sample; " + text).strip()[:700] or "aesthetic world sample"


def fuse_dense_vectors(weighted_vectors: list[tuple[list[float], float]]) -> list[float]:
    """Weighted L2-normalized vector fusion for visual and metadata embeddings."""

    valid = [(vector, max(0.0, float(weight))) for vector, weight in weighted_vectors if vector and weight > 0.0]
    if not valid:
        return []
    size = min(len(vector) for vector, _weight in valid)
    if size <= 0:
        return []
    accumulator = [0.0] * size
    total_weight = 0.0
    for vector, weight in valid:
        normalized = normalize_dense_vector(vector[:size])
        total_weight += weight
        for idx in range(size):
            accumulator[idx] += normalized[idx] * weight
    if total_weight <= 1e-9:
        return []
    return normalize_dense_vector([value / total_weight for value in accumulator])


def build_world_sample_index(samples_path: Path = WORLD_SAMPLES_PATH, batch_size: int = 8, max_items: int | None = None) -> dict[str, Any]:
    """Embed normalized samples and attach nearest-neighbor metadata."""

    samples = load_world_samples(samples_path, max_items=max_items)
    usable: list[tuple[dict[str, Any], Path]] = []
    for sample in samples:
        path = sample_asset_path(sample)
        if path and path.suffix.lower() in VALID_EXTENSIONS:
            usable.append((sample, path))

    items: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    for start in range(0, len(usable), max(1, batch_size)):
        batch = usable[start : start + batch_size]
        images: list[Image.Image] = []
        loaded_samples: list[dict[str, Any]] = []
        for sample, path in batch:
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            images.append(image)
            loaded_samples.append(sample)
        if not images:
            continue

        batch_vectors, status = try_siglip_embeddings(images)
        for image in images:
            image.close()
        if not batch_vectors:
            raise RuntimeError(f"Cannot build world sample index: {status}")

        signature_texts = [world_sample_signature_text(sample) for sample in loaded_samples]
        text_vectors, text_status = try_siglip_text_embeddings(signature_texts)
        if not text_vectors or len(text_vectors) != len(loaded_samples):
            text_vectors = [[] for _sample in loaded_samples]
            text_status = "metadata-text-unavailable"

        for sample, vector, text_vector, signature_text in zip(loaded_samples, batch_vectors, text_vectors, signature_texts):
            modalities = sample.get("modalities", {}) if isinstance(sample.get("modalities"), dict) else {}
            palette = sample.get("palette", []) if isinstance(sample.get("palette"), list) else []
            fused_vector = vector
            embedding_kind = "visual"
            if text_vector and len(text_vector) == len(vector):
                fused_vector = fuse_dense_vectors([(vector, 0.72), (text_vector, 0.28)]) or vector
                embedding_kind = "visual_metadata_fusion"
            items.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "dataset_origin": sample.get("dataset_origin"),
                    "asset_ref": sample.get("asset_ref"),
                    "palette": palette[:12],
                    "modalities": modalities,
                    "aesthetic_score": sample.get("aesthetic_score"),
                    "embedding": [round(float(value), 7) for value in fused_vector],
                    "embeddingKind": embedding_kind,
                    "textEmbeddingStatus": text_status,
                    "textSignature": signature_text,
                    "neighbors": [],
                }
            )
            embeddings.append(fused_vector)

    for idx, item in enumerate(items):
        ranked = []
        for other_idx, other in enumerate(items):
            if idx == other_idx:
                continue
            ranked.append(
                {
                    "sample_id": other.get("sample_id"),
                    "dataset_origin": other.get("dataset_origin"),
                    "similarity": round(cosine_similarity(embeddings[idx], embeddings[other_idx]), 5),
                }
            )
        ranked.sort(key=lambda value: value["similarity"], reverse=True)
        item["neighbors"] = ranked[:8]

    return {
        "version": "world-sample-index-v1",
        "schemaVersion": WORLD_SAMPLE_SCHEMA_VERSION,
        "analysisModelVersion": ANALYSIS_MODEL_VERSION,
        "embeddingModel": SIGLIP_MODEL_ID,
        "embeddingKind": "visual_metadata_fusion",
        "source": str(samples_path.relative_to(BASE_DIR)) if samples_path.is_relative_to(BASE_DIR) else str(samples_path),
        "updated": time.time(),
        "itemCount": len(items),
        "items": items,
    }
