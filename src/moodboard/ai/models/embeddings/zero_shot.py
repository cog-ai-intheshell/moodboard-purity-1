"""Zero-shot concept detection built on top of SigLIP text embeddings."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.moodboard.ai.models.embeddings.siglip2 import try_siglip_text_embeddings
from src.moodboard.ai.variables.labels import (
    OBSERVATION_FIELD_BY_TYPE,
    add_observation,
    clean_concept_label,
    make_observation,
    normalize_observation_type,
    rebuild_modalities_from_observations,
)
from src.moodboard.ai.variables.taxonomies import (
    ZERO_SHOT_CONCEPTS,
    build_world_taxonomy,
    load_aesthetic_knowledge,
    modality_prompt,
)
from src.moodboard.analysis.scoring import cosine_similarity, normalize_scores
from src.moodboard.core.schemas import clamp_float


def build_zero_shot_catalog() -> list[tuple[str, str, str]]:
    """Create the prompt catalog used for SigLIP2 zero-shot matching."""

    aesthetics = load_aesthetic_knowledge()
    catalog: list[tuple[str, str, str]] = []
    if len(aesthetics) < 20:
        catalog.extend(item for item in ZERO_SHOT_CONCEPTS if normalize_observation_type(item[1]) != "emotion")
    taxonomy = build_world_taxonomy(aesthetics, max_per_type=180)
    for obs_type in ("aesthetic", "symbol", "object", "texture", "style", "composition", "tag"):
        limit = 140 if obs_type in {"symbol", "style", "aesthetic"} else 80
        for value in taxonomy.get(obs_type, [])[:limit]:
            label = clean_concept_label(value, max_words=5)
            if label and len(label) <= 48:
                catalog.append((modality_prompt(label, obs_type), obs_type, label))

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for prompt, obs_type, value in catalog:
        key = (obs_type, value.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((prompt, obs_type, value))
        if len(out) >= 360:
            break
    return out


def apply_zero_shot_concepts(
    image_entries: list[dict[str, Any]],
    vectors: list[list[float]],
    emotion_counter: Counter[str],
    term_counter: Counter[str],
) -> str:
    """Attach zero-shot labels to image entries using existing image vectors."""

    if not vectors or not image_entries:
        return "no-vectors"
    catalog = build_zero_shot_catalog()
    prompts = [prompt for prompt, _bucket, _value in catalog]
    text_vectors, text_status = try_siglip_text_embeddings(prompts)
    if not text_vectors or len(text_vectors[0]) != len(vectors[0]):
        return text_status

    for idx, entry in enumerate(image_entries):
        scored = [(label_idx, cosine_similarity(vectors[idx], text_vector)) for label_idx, text_vector in enumerate(text_vectors)]
        if not scored:
            continue
        mean_score = sum(score for _label_idx, score in scored) / len(scored)
        variance = sum((score - mean_score) ** 2 for _label_idx, score in scored) / max(1, len(scored))
        std_score = math.sqrt(variance)
        threshold = mean_score + std_score * 0.72
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:10]
        zero_shot = []
        current_emotions = set(entry.get("emotionScores", {}).keys())
        for label_idx, similarity in ranked:
            if similarity < threshold and len(zero_shot) >= 3:
                continue
            prompt, bucket, value = catalog[label_idx]
            score = round(clamp_float((similarity + 1.0) / 2.0, 0.0, 0.0, 1.0), 4)
            zero_shot.append({"label": value, "prompt": prompt, "type": bucket, "score": score})
            obs_type = normalize_observation_type(bucket)
            add_observation(entry, make_observation(value, obs_type, score, "siglip2-zero-shot"))
            if obs_type == "emotion":
                if value not in current_emotions:
                    emotion_counter.update([value])
                    current_emotions.add(value)
                continue
            field = OBSERVATION_FIELD_BY_TYPE.get(obs_type, bucket)
            existing = set(entry.get(field, []))
            if field in {"tags", "symbols", "styles", "composition", "objects"} and value not in existing:
                entry[field] = sorted(existing | {value})
                term_counter.update([value])
        if current_emotions:
            entry["emotionScores"] = normalize_scores({emotion: 1.0 for emotion in current_emotions})
        rebuild_modalities_from_observations(entry)
        entry["zeroShot"] = zero_shot
    return text_status
