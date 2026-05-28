"""Shared label taxonomies and normalization helpers.

These utilities are intentionally model-agnostic: scrapers, caption adapters,
fusion and graph construction all need the same canonical modality names and
the same conservative label cleaner.
"""

from __future__ import annotations

import html as html_lib
import re
from typing import Any

from src.moodboard.analysis.scoring import normalize_scores
from src.moodboard.core.schemas import clamp_float


WORLD_TEXT_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "although", "among", "around", "and", "based",
    "because", "before", "being", "between", "called", "common", "commonly", "could", "culture",
    "design", "designs", "different", "early", "example", "examples", "focused", "from", "have",
    "image", "images", "including", "internet", "late", "like", "made", "media", "more", "most",
    "movement", "often", "page", "people", "present", "related", "style", "styles", "their",
    "there", "these", "they", "the", "this", "through", "used", "using", "visual", "visuals", "wiki",
    "with", "without", "would",
    "aesthetic", "aesthetics", "cari", "category", "categories", "fandom", "institute", "links",
    "official", "portuguese", "reddit", "references", "retrieved", "site", "website", "word",
}

OBSERVATION_FIELD_BY_TYPE = {
    "tag": "tags",
    "symbol": "symbols",
    "object": "objects",
    "texture": "textures",
    "style": "styles",
    "affect": "affects",
    "composition": "composition",
}

OBSERVATION_TYPE_ALIASES = {
    "tags": "tag",
    "symbols": "symbol",
    "styles": "style",
    "emotions": "emotion",
    "objects": "object",
    "textures": "texture",
    "aesthetics": "aesthetic",
    "affects": "affect",
    "affect_tags": "affect",
}


def observation_key(label: str, obs_type: str) -> str:
    """Build a stable id for a modality observation."""

    return re.sub(r"[^a-z0-9]+", "-", f"{obs_type}-{label}".lower()).strip("-")


def normalize_observation_type(value: str) -> str:
    """Map plural/API names to the canonical singular modality name."""

    return OBSERVATION_TYPE_ALIASES.get(value, value)


def make_observation(
    label: str,
    obs_type: str,
    confidence: float,
    source: str,
    *,
    bbox: list[float] | None = None,
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a normalized multimodal observation record."""

    clean_label = re.sub(r"\s+", " ", str(label or "").strip())
    canonical_type = normalize_observation_type(obs_type)
    return {
        "id": observation_key(clean_label, canonical_type),
        "label": clean_label,
        "type": canonical_type,
        "confidence": round(clamp_float(confidence, 0.0, 0.0, 1.0), 4),
        "source": source,
        "bbox": bbox,
        "embedding": embedding,
        "metadata": metadata or {},
    }


def add_observation(entry: dict[str, Any], observation: dict[str, Any]) -> None:
    """Merge an observation into an image entry, preserving best confidence."""

    label = str(observation.get("label", "")).strip()
    obs_type = normalize_observation_type(str(observation.get("type", "")).strip())
    observation["type"] = obs_type
    if not label or not obs_type:
        return
    observations = entry.setdefault("observations", [])
    obs_id = observation.get("id") or observation_key(label, obs_type)
    for existing in observations:
        if existing.get("id") == obs_id:
            if float(observation.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                existing.update(observation)
            else:
                sources = set(str(existing.get("source", "")).split("+"))
                sources.add(str(observation.get("source", "")))
                existing["source"] = "+".join(sorted(source for source in sources if source))
                if observation.get("bbox") and not existing.get("bbox"):
                    existing["bbox"] = observation.get("bbox")
                if observation.get("metadata"):
                    metadata = dict(existing.get("metadata", {}) or {})
                    metadata.update(observation.get("metadata", {}) or {})
                    existing["metadata"] = metadata
            return
    observations.append(observation)


def add_term_observations(entry: dict[str, Any], terms: list[str], obs_type: str, source: str, confidence: float) -> None:
    """Add a list of plain labels as observation records."""

    for term in terms:
        add_observation(entry, make_observation(term, obs_type, confidence, source))


def remove_observations_of_type(entry: dict[str, Any], obs_type: str) -> None:
    """Remove all observations of one canonical modality type from an entry."""

    entry["observations"] = [
        observation
        for observation in entry.get("observations", [])
        if str(observation.get("type", "")) != obs_type
    ]
    if obs_type == "emotion":
        entry["emotionScores"] = {}


def rebuild_modalities_from_observations(entry: dict[str, Any]) -> None:
    """Rebuild legacy modality fields from canonical observation records."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for observation in entry.get("observations", []):
        grouped.setdefault(str(observation.get("type", "")), []).append(observation)

    for obs_type, field in OBSERVATION_FIELD_BY_TYPE.items():
        items = grouped.get(obs_type, [])
        items.sort(key=lambda item: (float(item.get("confidence", 0.0)), str(item.get("label", ""))), reverse=True)
        entry[field] = [str(item.get("label")) for item in items if item.get("label")]

    emotion_scores = {
        str(item.get("label")).lower(): max(float(item.get("confidence", 0.0)), 0.05)
        for item in grouped.get("emotion", [])
        if item.get("label")
    }
    if emotion_scores:
        entry["emotionScores"] = normalize_scores(emotion_scores)


def clean_concept_label(value: str, max_words: int = 4) -> str:
    """Clean noisy scraped/caption labels while preserving useful concepts."""

    text = html_lib.unescape(str(value or "").lower())
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zÀ-ÿ0-9 '&/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,_-")
    text = re.sub(r"^(and|or|the|a|an)\s+", "", text)
    text = re.sub(r"\s+(and|or|the|a|an)$", "", text)
    words = text.split()
    compacted: list[str] = []
    for word in words:
        if compacted and compacted[-1] == word:
            continue
        compacted.append(word)
    text = " ".join(compacted)
    bad_fragments = (
        "aesthetics wiki",
        "cari institute",
        "category ",
        "external links",
        "facebook group",
        "official website",
        "portuguese word",
        "reddit community",
        "references",
        "retrieved from",
        "the word for",
        "word for",
    )
    if any(fragment in text for fragment in bad_fragments):
        return ""
    words = [word.strip("'&/-") for word in text.split() if word.strip("'&/-")]
    words = [word for word in words if word not in WORLD_TEXT_STOPWORDS and len(word) > 2]
    if not words or len(words) > max_words:
        return ""
    label = " ".join(words)
    if any(char.isdigit() for char in label) and not re.search(r"\b(2d|3d|4k|8k|90s|00s|2000s|2010s|2020s)\b", label):
        return ""
    return label

SYMBOLIC_LABEL_GROUPS = (
    "mythic",
    "technological",
    "heroic",
    "mortality",
    "transgression",
    "sacred",
    "domestic",
    "organic",
)

AFFECT_LABEL_GROUPS = (
    "calm",
    "nostalgia",
    "tension",
    "melancholy",
    "awe",
    "intimacy",
    "alienation",
    "energy",
)
