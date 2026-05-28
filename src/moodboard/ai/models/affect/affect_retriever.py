"""Affective and symbolic-affect retrieval adapter.

The affect model uses the current aesthetic corpus to build an affective
vocabulary, embeds those prompts with SigLIP2 text embeddings, then attaches the
closest emotional or symbolic-value observations to each image vector.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

from src.moodboard.ai.models.embeddings.siglip2 import try_siglip_text_embeddings
from src.moodboard.ai.variables.labels import (
    add_observation,
    clean_concept_label,
    make_observation,
    rebuild_modalities_from_observations,
    remove_observations_of_type,
)
from src.moodboard.analysis.scoring import cosine_similarity
from src.moodboard.core.schemas import clamp_float


AFFECTIVE_VALUE_TERMS = {
    "alienation", "chaos", "devotion", "escapism", "existential dread", "heroism",
    "liminality", "mortality", "mysticism", "nihilism", "otherworldliness", "rebellion",
    "purity", "sacredness", "spirituality", "tech-optimism", "tech optimism",
    "techno-optimism", "techno optimism", "transgression",
    "violence",
}


def derive_affective_vocabulary(aesthetics: list[dict[str, Any]], max_terms: int = 96) -> list[str]:
    """Derive a high-signal affect vocabulary from aesthetic metadata."""

    counter: Counter[str] = Counter()
    base_affective_terms = [
        "melancholy", "nostalgia", "serenity", "awe", "dread", "unease", "tenderness", "intimacy",
        "isolation", "longing", "wonder", "menace", "tension", "calm", "grief", "joy", "euphoria",
        "mystery", "alienation", "innocence", "romance", "anxiety", "solemnity", "reverence",
        "sensuality", "comfort", "chaos", "violence", "rebellion", "escapism", "vulnerability",
        "confidence", "optimism", "nihilism", "dreaminess", "uncanny", "mysticism", "spirituality",
        "majesty", "loneliness", "warmth", "coldness", "fragility", "devotion", "desire",
        "tranquility", "suspense", "disorientation", "rapture", "mourning", "detachment",
        "playfulness", "hope", "fear", "rage", "softness", "severity", "sacredness",
        "heroism", "mortality", "transgression", "techno-optimism", "otherworldliness",
    ]
    affect_roots = {
        "alien", "anx", "awe", "calm", "chaos", "comfort", "cold", "desire", "detachment",
        "devotion", "disorient", "dread", "dream", "euphor", "escap", "fear", "fragil",
        "grief", "hope", "innocen", "intim", "isolat", "joy", "long", "melanchol",
        "menace", "mourn", "myster", "mystic", "nihil", "nostalg", "optim", "play",
        "rage", "raptur", "rebel", "rever", "romanc", "sacred", "sensual", "seren",
        "sever", "soft", "solemn", "spirit", "suspense", "tender", "tension",
        "tranquil", "uncanny", "unease", "violence", "vulner", "warm", "wonder",
    }
    stop = {
        "aesthetic", "aesthetics", "style", "styles", "visual", "visuals", "image", "images", "color",
        "colors", "design", "designs", "often", "common", "commonly", "related", "media", "based",
        "with", "from", "into", "that", "this", "their", "through", "around", "used", "using",
        "early", "late", "mid", "present", "century", "internet", "culture", "movement",
        "black", "white", "blue", "green", "red", "pink", "purple", "yellow", "orange", "brown",
        "gray", "grey", "cyan", "gold", "silver", "pastel", "neon", "dark", "light", "soft",
        "photography", "painting", "digital", "vector", "architecture", "fashion", "music",
        "called", "focused", "evolved", "graphic", "graphics", "artistic", "futurism", "examples",
        "example", "pending", "research", "website", "http", "https", "www", "are", "cari",
        "institute", "consumer", "evan", "collins", "facebook", "group",
        "creator", "created", "gothic", "city", "cities", "forest", "water", "screen", "book",
        "books", "architecture", "object", "objects", "symbol", "symbols", "motif", "motifs",
        "uniform", "uniforms", "imagery", "goods", "girl", "girls", "boy", "boys", "sailor",
        "magical", "weapon", "weapons", "clothing", "outfit", "outfits",
        "beauty", "kindness", "goodness", "craftsmanship", "sophistication", "selfie",
        "awareness", "sufficiency", "consumerism", "luxury", "status", "wealth",
        "tech", "technology",
    }
    bad_fragments = {
        "selfie", "craftsmanship", "consumer", "sophistication", "self-awareness", "self-sufficiency",
        "goodness", "kindness", "beauty standard", "luxury", "brand", "fashion trend",
        "technology",
    }
    abstract_suffixes = (
        "ness", "ity", "ism", "tion", "sion", "ment", "ance", "ence", "ship", "dom",
        "cy", "ethos", "pathy", "philia", "phobia", "algia", "dread", "core",
    )
    compact_affect_terms = set(base_affective_terms) | {"liminality", "curiosity", "irony", "hedonism"}

    def clean_affect_term(value: str) -> str:
        term = re.sub(r"https?://\S+", " ", str(value).lower())
        term = re.sub(r"www\.\S+", " ", term)
        term = re.sub(r"[^a-zÀ-ÿ0-9 -]+", " ", term)
        term = re.sub(r"\s+", " ", term).strip(" .:_-")
        term = re.sub(r"^(and|or)\s+", "", term)
        term = re.sub(r"\s+(and|or)$", "", term)
        return term

    def is_quality_affect_term(term: str) -> bool:
        if not (3 <= len(term) <= 34):
            return False
        if any(fragment in term for fragment in bad_fragments):
            return False
        if any(char.isdigit() for char in term) or "." in term:
            return False
        words = term.split()
        if len(words) > 3:
            return False
        if any(word in stop or len(word) < 3 for word in words):
            return False
        if len(words) == 1:
            word = words[0]
            return word in compact_affect_terms or word.endswith(abstract_suffixes)
        if any(word.startswith(("self", "anti", "pro")) for word in words):
            return False
        if not any(any(root in word for root in affect_roots) for word in words):
            return False
        if any(word in compact_affect_terms or word.endswith(abstract_suffixes) for word in words):
            return True
        return False

    for term in base_affective_terms:
        counter[term] += 8

    for item in aesthetics:
        for value in item.get("emotions", []) or []:
            term = clean_affect_term(str(value))
            if is_quality_affect_term(term):
                counter[term] += 5
        value_text = str(item.get("values", ""))
        for value in re.split(r",|;|/|\n|\|", value_text):
            term = clean_affect_term(value)
            if is_quality_affect_term(term):
                counter[term] += 3

    terms = []
    seen: set[str] = set()
    for term, _amount in counter.most_common(max_terms * 3):
        clean = re.sub(r"\s+", " ", term).strip()
        if not clean or clean in seen:
            continue
        if any(existing in clean or clean in existing for existing in seen if abs(len(existing) - len(clean)) < 5):
            continue
        seen.add(clean)
        terms.append(clean)
        if len(terms) >= max_terms:
            break
    for term in sorted(AFFECTIVE_VALUE_TERMS):
        clean = re.sub(r"\s+", " ", term).strip()
        if clean and clean not in seen:
            seen.add(clean)
            terms.append(clean)
    return terms


def affective_observation_type(label: str) -> str:
    """Route abstract values to `affect` and felt moods to `emotion`."""

    clean = clean_concept_label(label, max_words=4)
    if clean in AFFECTIVE_VALUE_TERMS:
        return "affect"
    words = clean.split()
    if any(word in {"heroism", "mortality", "transgression", "nihilism", "spirituality", "mysticism"} for word in words):
        return "affect"
    if clean.endswith(("ism", "ality")) and clean not in {"optimism"}:
        return "affect"
    if "dread" in words and len(words) > 1:
        return "affect"
    if clean.startswith("techno"):
        return "affect"
    return "emotion"


def canonical_affective_label(label: str) -> str:
    """Normalize equivalent symbolic-affective labels."""

    clean = clean_concept_label(label, max_words=4)
    if clean in {"tech optimism", "tech-optimism", "techno optimism"}:
        return "techno-optimism"
    return clean


def apply_affective_model(
    image_entries: list[dict[str, Any]],
    vectors: list[list[float]],
    aesthetics: list[dict[str, Any]],
    context_matches: list[dict[str, Any]] | None = None,
) -> str:
    """Attach affect/emotion observations from corpus-derived text prompts."""

    if not vectors or not image_entries:
        return "no-vectors"
    if context_matches:
        by_name = {str(item.get("name", "")).lower(): item for item in aesthetics}
        contextual_items = [
            by_name[str(match.get("name", "")).lower()]
            for match in context_matches[:12]
            if str(match.get("name", "")).lower() in by_name
        ]
    else:
        contextual_items = []
    vocabulary = derive_affective_vocabulary(contextual_items, 72) if contextual_items else []
    if len(vocabulary) < 24:
        for term in derive_affective_vocabulary(aesthetics, 96):
            if term not in vocabulary:
                vocabulary.append(term)
            if len(vocabulary) >= 96:
                break
    if not vocabulary:
        return "no-affective-vocabulary"
    prompts = [
        f"an image that evokes {term} as an aesthetic mood, atmosphere, feeling or emotional value"
        for term in vocabulary
    ]
    text_vectors, text_status = try_siglip_text_embeddings(prompts)
    if not text_vectors or len(text_vectors[0]) != len(vectors[0]):
        return text_status

    for idx, entry in enumerate(image_entries):
        remove_observations_of_type(entry, "emotion")
        scored = [(label_idx, cosine_similarity(vectors[idx], text_vector)) for label_idx, text_vector in enumerate(text_vectors)]
        if not scored:
            rebuild_modalities_from_observations(entry)
            continue
        mean_score = sum(score for _label_idx, score in scored) / len(scored)
        variance = sum((score - mean_score) ** 2 for _label_idx, score in scored) / max(1, len(scored))
        threshold = mean_score + math.sqrt(variance) * 0.82
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:8]
        selected = []
        for rank, (label_idx, similarity) in enumerate(ranked):
            confidence = clamp_float((similarity + 1.0) / 2.0, 0.0, 0.0, 1.0)
            if rank < 2 or similarity >= threshold:
                selected.append((vocabulary[label_idx], confidence))
            if len(selected) >= 5:
                break
        for label, confidence in selected:
            canonical_label = canonical_affective_label(label)
            obs_type = affective_observation_type(canonical_label)
            add_observation(entry, make_observation(canonical_label, obs_type, confidence, "siglip2-corpus-affect"))
        rebuild_modalities_from_observations(entry)
    return f"{text_status}; corpus-affect vocabulary={len(vocabulary)}"


def apply(entries: list[dict[str, Any]], vectors: list[list[float]], aesthetics: list[dict[str, Any]]) -> str:
    """Registry-friendly alias for affect retrieval."""

    return apply_affective_model(entries, vectors, aesthetics)
