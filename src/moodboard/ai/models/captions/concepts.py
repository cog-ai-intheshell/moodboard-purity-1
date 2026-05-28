"""Caption-to-modality concept extraction.

Caption models return free text, but the graph needs typed nodes: objects,
symbols, textures, styles, composition tags and affective concepts. This module
keeps that translation next to the caption adapters instead of burying it in
the high-level moodboard analyzer.
"""

from __future__ import annotations

import html as html_lib
import re
from collections import Counter
from typing import Any

from src.moodboard.ai.models.embeddings.siglip2 import try_siglip_text_embeddings
from src.moodboard.ai.variables.labels import (
    OBSERVATION_FIELD_BY_TYPE,
    WORLD_TEXT_STOPWORDS,
    clean_concept_label,
    normalize_observation_type,
)
from src.moodboard.ai.variables.taxonomies import (
    COMPOSITION_MARKERS,
    MODALITY_PROMPTS,
    OBJECT_MARKERS,
    STYLE_ADJECTIVE_MARKERS,
    TEXTURE_MARKERS,
)
from src.moodboard.analysis.scoring import cosine_similarity
from src.moodboard.core.schemas import clamp_float


CAPTION_STOPWORDS = WORLD_TEXT_STOPWORDS | {
    "appears",
    "background",
    "close",
    "close-up",
    "front",
    "hands",
    "has",
    "holding",
    "image",
    "long",
    "made",
    "middle",
    "photo",
    "photograph",
    "picture",
    "shows",
    "sitting",
    "slender",
    "standing",
    "their",
    "wearing",
    "blurred",
    "depiction",
    "depicts",
    "surrounded",
}


def extract_caption_candidates(caption: str, max_terms: int = 18) -> list[str]:
    """Extract compact noun/phrase candidates from one generated caption."""

    text = html_lib.unescape(str(caption or "").lower())
    text = re.sub(r"[^a-zÀ-ÿ0-9 -]+", " ", text)
    tokens = [
        token
        for token in re.findall(r"[a-zÀ-ÿ][a-zÀ-ÿ0-9-]{2,}", text)
        if token not in CAPTION_STOPWORDS
    ]
    counter: Counter[str] = Counter()
    for size in (3, 2):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            phrase = clean_concept_label(" ".join(tokens[idx : idx + size]), max_words=3)
            if phrase:
                counter[phrase] += size + 1
    for token in tokens:
        clean = clean_concept_label(token, max_words=1)
        if clean:
            counter[clean] += 1

    labels = []
    for term, _amount in counter.most_common(max_terms * 2):
        # Avoid visually noisy duplicates such as "black armor" and "armor".
        if any(term in existing or existing in term for existing in labels if abs(len(existing) - len(term)) < 8):
            continue
        labels.append(term)
        if len(labels) >= max_terms:
            break
    return labels


def classify_terms_by_modality(terms: list[str]) -> dict[str, dict[str, Any]]:
    """Classify caption terms into the graph's canonical modalities."""

    clean_terms = [clean_concept_label(term, max_words=4) for term in terms]
    clean_terms = [term for term in clean_terms if term]
    if not clean_terms:
        return {}

    prototype_items = list(MODALITY_PROMPTS.items())
    prototype_texts = [f"this phrase describes {prompt}" for _bucket, prompt in prototype_items]
    term_texts = [f"visual phrase: {term}" for term in clean_terms]
    vectors, status = try_siglip_text_embeddings(prototype_texts + term_texts)
    if not vectors or len(vectors) != len(prototype_texts) + len(term_texts):
        return {
            term: {"type": "tag", "confidence": 0.46, "source": f"caption-term-fallback:{status}"}
            for term in clean_terms
        }

    prototype_vectors = vectors[: len(prototype_texts)]
    term_vectors = vectors[len(prototype_texts) :]
    classified: dict[str, dict[str, Any]] = {}
    for term, vector in zip(clean_terms, term_vectors):
        scores = [
            (bucket, cosine_similarity(vector, prototype_vector))
            for (bucket, _prompt), prototype_vector in zip(prototype_items, prototype_vectors)
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        best_bucket, best_score = scores[0]

        # Text embeddings handle many terms, but these tiny lexical guards keep
        # common material/object words from being mislabeled as broad styles.
        texture_hit = any(marker in term for marker in TEXTURE_MARKERS)
        object_hit = any(marker == term or marker in term.split() or marker in term for marker in OBJECT_MARKERS)
        style_adjective_hit = any(marker == term or marker in term.split() for marker in STYLE_ADJECTIVE_MARKERS)
        if object_hit and best_bucket in {"symbol", "tag", "style", "composition"} and not style_adjective_hit:
            object_score = next((score for bucket, score in scores if bucket == "object"), best_score)
            best_bucket, best_score = "object", object_score
        if texture_hit and best_bucket in {"symbol", "tag", "texture"}:
            texture_score = next((score for bucket, score in scores if bucket == "texture"), best_score)
            best_bucket, best_score = "texture", texture_score
        if best_bucket == "texture" and not texture_hit:
            non_texture = next((item for item in scores if item[0] != "texture"), None)
            if non_texture:
                best_bucket, best_score = non_texture
        if best_bucket == "composition" and not any(marker in term for marker in COMPOSITION_MARKERS) and len(term.split()) <= 2:
            non_composition = next((item for item in scores if item[0] not in {"composition", "texture"}), None)
            if non_composition:
                best_bucket, best_score = non_composition

        second_score = scores[1][1] if len(scores) > 1 else best_score
        confidence = clamp_float(0.50 + ((best_score - second_score) * 0.9) + ((best_score + 1.0) * 0.08), 0.48, 0.0, 0.86)
        classified[term] = {
            "type": best_bucket,
            "confidence": round(confidence, 4),
            "source": "siglip2-caption-modality",
            "scores": {bucket: round((score + 1.0) / 2.0, 4) for bucket, score in scores[:3]},
        }
    return classified


def canonical_texture_label(term: str) -> str:
    """Reduce texture phrases to a stable graph label when possible."""

    words = re.findall(r"[a-zÀ-ÿ][a-zÀ-ÿ0-9-]{2,}", str(term or "").lower())
    for word in words:
        if word in TEXTURE_MARKERS:
            return word
    for word in words:
        for marker in TEXTURE_MARKERS:
            if marker in word:
                return word
    return clean_concept_label(term, max_words=3)


def canonical_object_label(term: str) -> str:
    """Reduce object phrases to known object markers while preserving specifics."""

    clean = clean_concept_label(term, max_words=4)
    words = clean.split()
    for marker in sorted(OBJECT_MARKERS, key=len, reverse=True):
        marker_words = marker.split()
        if len(marker_words) > 1 and marker in clean:
            return marker
        if len(marker_words) == 1 and marker in words:
            return marker
    if len(words) <= 2:
        return clean
    return ""


def caption_object_candidates(caption: str, classified: dict[str, dict[str, Any]], limit: int = 10) -> list[str]:
    """Recover object nodes that generic caption parsing might underweight."""

    candidates: list[str] = []
    for term in extract_caption_candidates(caption, max_terms=32):
        clean = clean_concept_label(term, max_words=4)
        if not clean or clean in CAPTION_STOPWORDS:
            continue
        words = clean.split()
        if any(word in STYLE_ADJECTIVE_MARKERS for word in words) and not any(word in OBJECT_MARKERS for word in words):
            continue
        canonical = canonical_object_label(clean)
        if any(marker == clean or marker in words or marker in clean for marker in OBJECT_MARKERS):
            if canonical:
                candidates.append(canonical)
            continue
        payload = classified.get(clean, {})
        scores = payload.get("scores", {}) if isinstance(payload, dict) else {}
        object_score = float(scores.get("object", 0.0) or 0.0)
        if len(words) <= 2 and object_score >= 0.515 and not any(word in TEXTURE_MARKERS for word in words):
            candidates.append(canonical or clean)

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if any(candidate in existing or existing in candidate for existing in out if abs(len(existing) - len(candidate)) <= 8):
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


def caption_concepts(caption: str) -> dict[str, list[str]]:
    """Convert one caption into typed concept buckets used by the graph."""

    buckets: dict[str, set[str]] = {
        "tags": set(),
        "objects": set(),
        "symbols": set(),
        "textures": set(),
        "styles": set(),
        "composition": set(),
        "emotions": set(),
    }
    candidates = extract_caption_candidates(caption)
    classified = classify_terms_by_modality(candidates)
    for term, payload in classified.items():
        obs_type = normalize_observation_type(str(payload.get("type", "tag")))
        field = OBSERVATION_FIELD_BY_TYPE.get(obs_type, "tags")
        if field == "textures":
            output_term = canonical_texture_label(term)
        elif field == "objects":
            output_term = canonical_object_label(term)
        else:
            output_term = term
        if not output_term:
            continue
        if field in buckets:
            buckets[field].add(output_term)
        else:
            buckets["tags"].add(output_term)
    for term in caption_object_candidates(caption, classified):
        buckets["objects"].add(term)
    return {key: sorted(values) for key, values in buckets.items()}
