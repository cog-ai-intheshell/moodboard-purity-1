"""Builders and loaders for the local aesthetic text-embedding index.

The runtime can match moodboards against the aesthetic knowledge base without
training at startup. This module is used by offline world-learning jobs to
precompute the index once, and by analysis code to load the resulting artifact.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.moodboard.ai.models.embeddings.siglip2 import try_siglip_text_embeddings
from src.moodboard.ai.registry import ANALYSIS_MODEL_VERSION, SIGLIP_MODEL_ID
from src.moodboard.ai.variables.taxonomies import aesthetic_keywords, load_aesthetic_knowledge
from src.moodboard.analysis.scoring import cosine_similarity
from src.moodboard.core.paths import AESTHETIC_INDEX_PATH
from src.moodboard.core.schemas import clamp_float


def read_json_file(path: Path) -> dict[str, Any] | None:
    """Read a JSON object artifact, returning ``None`` when unavailable."""

    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception as exc:
        print(f"[WARN] Cannot read JSON artifact {path}: {exc}", file=sys.stderr)
    return None


def aesthetic_index_source_text(item: dict[str, Any]) -> str:
    """Convert one aesthetic record into the text embedded by SigLIP2."""

    parts = [
        str(item.get("name", "")),
        str(item.get("description", "")),
        "colors " + " ".join(str(value) for value in item.get("colors", []) if value),
        "emotions " + " ".join(str(value) for value in item.get("emotions", []) if value),
        "symbols " + " ".join(str(value) for value in item.get("symbols", []) if value),
        "styles " + " ".join(str(value) for value in item.get("styles", []) if value),
        "tags " + " ".join(str(value) for value in item.get("tags", []) if value),
        "related " + " ".join(str(value) for value in item.get("related", []) if value),
    ]
    return " ".join(" ".join(parts).split()[:96])


def load_aesthetic_embedding_index(path: Path = AESTHETIC_INDEX_PATH) -> dict[str, Any] | None:
    """Load the precomputed aesthetic index if it matches the active encoder."""

    payload = read_json_file(path)
    if not payload:
        return None
    if payload.get("embeddingModel") != SIGLIP_MODEL_ID:
        return None
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return None
    return payload


def match_aesthetics(
    image_entries: list[dict[str, Any]],
    global_terms: Counter[str],
    global_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Rank known aesthetics against extracted terms and optional embeddings."""

    aesthetics = load_aesthetic_knowledge()
    matches: list[dict[str, Any]] = []
    mood_terms = set(global_terms.keys())
    color_terms = {str(color["name"]).lower() for entry in image_entries for color in entry.get("palette", [])}
    retrieval_terms = mood_terms | color_terms
    keyword_cache = {str(item.get("name", "")): aesthetic_keywords(item) for item in aesthetics}

    def prefilter_score(item: dict[str, Any]) -> float:
        keywords = keyword_cache.get(str(item.get("name", "")), set())
        overlap = len(retrieval_terms & keywords)
        explicit = 0
        for field in ("colors", "emotions", "symbols", "styles", "tags", "related"):
            explicit += len({str(value).lower() for value in item.get(field, [])} & retrieval_terms)
        return overlap * 0.4 + explicit * 1.0 + min(1.0, len(str(item.get("description", ""))) / 900.0) * 0.15

    vector_scores: dict[str, float] = {}
    if global_vector:
        trained_index = load_aesthetic_embedding_index()
        if trained_index:
            for item in trained_index.get("items", []):
                vector = item.get("embedding", [])
                if isinstance(vector, list) and len(vector) == len(global_vector):
                    vector_scores[str(item.get("name", ""))] = clamp_float((cosine_similarity(global_vector, vector) + 1.0) / 2.0, 0.5, 0.0, 1.0)
        else:
            # Build only a small temporary text batch at runtime if the offline
            # index is missing; the world-learning job should normally persist it.
            candidate_items = sorted(aesthetics, key=prefilter_score, reverse=True)[:420]
            texts = [aesthetic_index_source_text(item) for item in candidate_items]
            text_vectors, _text_status = try_siglip_text_embeddings(texts)
            if text_vectors and len(text_vectors[0]) == len(global_vector):
                for item, text_vector in zip(candidate_items, text_vectors):
                    vector_scores[str(item.get("name", ""))] = clamp_float((cosine_similarity(global_vector, text_vector) + 1.0) / 2.0, 0.5, 0.0, 1.0)

    for item in aesthetics:
        keywords = keyword_cache.get(str(item.get("name", "")), aesthetic_keywords(item))
        overlap = len(retrieval_terms & keywords)
        explicit = 0
        for field in ("colors", "emotions", "symbols", "styles", "tags", "related"):
            explicit += len({str(value).lower() for value in item.get(field, [])} & retrieval_terms)
        description_bonus = min(8, overlap) * 0.055
        explicit_bonus = explicit * 0.11
        keyword_score = clamp_float(0.08 + description_bonus + explicit_bonus, 0.0, 0.0, 0.98)
        vector_score = vector_scores.get(str(item.get("name", "")))
        score = (keyword_score * 0.35 + vector_score * 0.65) if vector_score is not None else keyword_score
        matches.append(
            {
                "name": item.get("name", "Unknown"),
                "score": round(score, 4),
                "description": str(item.get("description", ""))[:220],
                "source": item.get("source", "local seed"),
                "vectorScore": round(vector_score, 4) if vector_score is not None else None,
                "keywords": sorted(keywords)[:32],
            }
        )
    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:6]


def build_aesthetic_embedding_index(batch_size: int = 64) -> dict[str, Any]:
    """Precompute text embeddings for every known aesthetic entry."""

    aesthetics = load_aesthetic_knowledge()
    items = []
    for item in aesthetics:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        items.append(
            {
                "name": name,
                "source": item.get("source", "unknown"),
                "description": str(item.get("description", ""))[:260],
                "keywords": sorted(aesthetic_keywords(item))[:48],
                "text": aesthetic_index_source_text(item),
            }
        )

    vectors: list[list[float]] = []
    for start in range(0, len(items), max(1, batch_size)):
        batch_texts = [item["text"] for item in items[start : start + batch_size]]
        batch_vectors, status = try_siglip_text_embeddings(batch_texts)
        if not batch_vectors:
            raise RuntimeError(f"Cannot build aesthetic index: {status}")
        vectors.extend(batch_vectors)

    for item, vector in zip(items, vectors):
        item["embedding"] = [round(float(value), 7) for value in vector]

    return {
        "version": "aesthetic-text-index-v1",
        "analysisModelVersion": ANALYSIS_MODEL_VERSION,
        "embeddingModel": SIGLIP_MODEL_ID,
        "updated": time.time(),
        "itemCount": len(items),
        "items": items,
    }
