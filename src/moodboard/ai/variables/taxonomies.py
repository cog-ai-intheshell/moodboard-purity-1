"""Aesthetic seed data, corpus loading and taxonomy builders.

Large/generated databases still live in `database/`. This module owns the
stable local seed and the normalization logic that turns scraped aesthetics into
taxonomies shared by zero-shot labels, OWLv2 grounding and affect retrieval.
"""

from __future__ import annotations

from collections import Counter
import json
import re
import sys
import time
from typing import Any

from scraping.aesthetic_sources import scrape_aesthetic_source_items
from src.moodboard.ai.variables.labels import WORLD_TEXT_STOPWORDS, clean_concept_label
from src.moodboard.core.paths import AESTHETICS_CACHE_PATH, DATABASE_DIR


TAXONOMY_KEYS = (
    "objects",
    "symbols",
    "textures",
    "styles",
    "emotions",
    "affects",
    "composition",
    "colors",
    "aesthetics",
)

AESTHETIC_SEED_DATA = [
    {
        "name": "Ethereal",
        "description": "Airy, luminous, delicate, spiritual and weightless visual atmosphere.",
        "colors": ["ivory", "foggy blue", "soft gold", "pastel pink"],
        "emotions": ["serenity", "spirituality", "mysticism", "innocence"],
        "symbols": ["halo", "light", "clouds", "wings", "veil"],
        "styles": ["soft fantasy", "watercolor", "glow"],
        "related": ["Angelcore", "Fairycore", "Cosmic Fantasy"],
    },
    {
        "name": "Dreamcore",
        "description": "Surreal, nostalgic, liminal and hazy images with uncanny softness.",
        "colors": ["pastel pink", "foggy blue", "lavender", "cream"],
        "emotions": ["nostalgia", "strangeness", "melancholy", "innocence"],
        "symbols": ["clouds", "eyes", "hallways", "stars", "rooms"],
        "styles": ["soft focus", "surreal", "liminal"],
        "related": ["Weirdcore", "Traumacore", "Ethereal"],
    },
    {
        "name": "Angelcore",
        "description": "Holy, soft, celestial and devotional atmosphere around light and purity.",
        "colors": ["white", "gold", "baby blue", "pale pink"],
        "emotions": ["spirituality", "serenity", "innocence", "majesty"],
        "symbols": ["halo", "wings", "light", "church", "dove"],
        "styles": ["sacred", "soft fantasy", "glow"],
        "related": ["Ethereal", "Royalcore", "Fairycore"],
    },
    {
        "name": "Cosmic Fantasy",
        "description": "Mystical space fantasy with stars, deep blues, purple light and scale.",
        "colors": ["deep purple", "midnight blue", "violet", "silver"],
        "emotions": ["mysticism", "majesty", "strangeness", "spirituality"],
        "symbols": ["stars", "cosmos", "moon", "spiral", "light"],
        "styles": ["digital painting", "soft fantasy", "glow"],
        "related": ["Ethereal", "Cosmic Horror", "Dreamcore"],
    },
    {
        "name": "Royalcore",
        "description": "Regal, ornate, historic and ceremonial visuals with status symbolism.",
        "colors": ["gold", "burgundy", "ivory", "emerald"],
        "emotions": ["majesty", "spirituality", "nostalgia"],
        "symbols": ["crown", "lion", "castle", "sword", "jewels"],
        "styles": ["ornamental", "oil painting", "historic"],
        "related": ["Angelcore", "Dark Academia", "Mythic Naturalism"],
    },
    {
        "name": "Dark Academia",
        "description": "Scholarly, old-world, melancholic and shadowed classical imagery.",
        "colors": ["brown", "black", "cream", "deep green"],
        "emotions": ["melancholy", "nostalgia", "mysticism"],
        "symbols": ["books", "architecture", "candles", "statues"],
        "styles": ["photography", "engraving", "classical"],
        "related": ["Royalcore", "Soft Gothic", "Light Academia"],
    },
    {
        "name": "Soft Gothic",
        "description": "Romantic darkness, muted contrast, melancholy and delicate tragedy.",
        "colors": ["black", "burgundy", "dusty pink", "gray"],
        "emotions": ["melancholy", "mysticism", "nostalgia", "strangeness"],
        "symbols": ["roses", "cross", "moon", "veil", "wounds"],
        "styles": ["ink", "romantic", "gothic"],
        "related": ["Dark Academia", "Traumacore", "Angelcore"],
    },
    {
        "name": "Traumacore",
        "description": "Fragile, tragic and uncanny imagery with childhood and symbolic pain.",
        "colors": ["pastel pink", "washed blue", "white", "red"],
        "emotions": ["melancholy", "strangeness", "violence", "innocence"],
        "symbols": ["wounds", "eyes", "toys", "rooms", "blood"],
        "styles": ["surreal", "soft focus", "collage"],
        "related": ["Dreamcore", "Weirdcore", "Soft Gothic"],
    },
    {
        "name": "Fairycore",
        "description": "Organic, natural, delicate and magical forest-adjacent visuals.",
        "colors": ["sage green", "moss", "pastel pink", "cream"],
        "emotions": ["serenity", "innocence", "mysticism"],
        "symbols": ["flowers", "mushrooms", "wings", "forest", "sparkles"],
        "styles": ["watercolor", "organic", "soft fantasy"],
        "related": ["Ethereal", "Angelcore", "Cottagecore"],
    },
    {
        "name": "Cyberpunk",
        "description": "Neon, urban, technological, saturated and high-contrast future decay.",
        "colors": ["cyan", "magenta", "black", "neon purple"],
        "emotions": ["chaos", "violence", "strangeness", "majesty"],
        "symbols": ["city", "machine", "neon", "weapon", "screen"],
        "styles": ["digital painting", "photography", "high contrast"],
        "related": ["Vaporwave", "Techwear", "Dystopian"],
    },
    {
        "name": "Frutiger Aero",
        "description": "Glossy optimistic 2000s tech, water, sky, glass and clean gradients.",
        "colors": ["cyan", "lime", "white", "sky blue"],
        "emotions": ["serenity", "innocence", "nostalgia"],
        "symbols": ["water", "glass", "sky", "bubbles", "leaf"],
        "styles": ["glossy", "vector", "digital"],
        "related": ["Y2K", "Webcore", "Eco Futurism"],
    },
    {
        "name": "Cosmic Horror",
        "description": "Vast, unknowable, dark cosmic scale with dread and strange symbols.",
        "colors": ["black", "deep purple", "green", "red"],
        "emotions": ["strangeness", "mysticism", "chaos", "violence"],
        "symbols": ["eyes", "spiral", "tentacles", "stars", "void"],
        "styles": ["dark fantasy", "ink", "high contrast"],
        "related": ["Cosmic Fantasy", "Weirdcore", "Dark Fantasy"],
    },
]

ZERO_SHOT_CONCEPTS = [
    ("pastel dreamlike", "tags", "pastel"),
    ("dark gothic", "styles", "gothic"),
    ("ethereal luminous", "tags", "ethereal"),
    ("cosmic mystical", "symbols", "cosmology"),
    ("angelic sacred halo", "symbols", "halo"),
    ("melancholic nostalgic", "emotions", "melancholy"),
    ("serene peaceful", "emotions", "serenity"),
    ("violent red dramatic", "emotions", "violence"),
    ("royal ornate", "styles", "ornamental"),
    ("organic natural forest", "symbols", "forest"),
    ("surreal liminal", "styles", "surreal"),
    ("neon cyberpunk", "styles", "high contrast"),
    ("watercolor soft fantasy", "styles", "watercolor"),
    ("digital painting", "styles", "digital painting"),
    ("photographic realistic", "styles", "photography"),
    ("minimal negative space", "composition", "negative space"),
    ("dense detailed composition", "composition", "dense"),
    ("vertical portrait composition", "composition", "verticality"),
    ("horizontal landscape composition", "composition", "horizontal flow"),
    ("central balanced composition", "composition", "central balance"),
]

MODALITY_PROMPTS = {
    "object": "a concrete visible object, person, animal, clothing, architecture or physical thing",
    "symbol": "a symbolic motif, icon, archetype, religious sign, mythic sign or recurring visual metaphor",
    "texture": "a material texture, surface, fabric, grain, metal, stone, paper, fog or tactile visual quality",
    "style": "an art style, visual medium, design direction, genre, rendering technique or fashion language",
    "affect": "a symbolic affect, value, mythic theme, cultural tension, worldview or existential meaning",
    "composition": "a composition, framing, layout, camera angle, symmetry, perspective, density or negative space pattern",
    "tag": "a general visual descriptor, atmosphere, theme or semantic concept",
}

TEXTURE_MARKERS = {
    "chrome", "dust", "fabric", "fog", "fur", "glass", "gloss", "glossy", "grain", "grainy",
    "ice", "ink", "lace", "matte", "metal", "metallic", "mist", "paper", "plastic", "silk",
    "snow", "snowy", "stone", "velvet", "water", "wood",
}

OBJECT_MARKERS = {
    "armor", "balustrade", "bird", "blackboard", "blade", "book", "building", "cathedral",
    "chair", "chalkboard", "dress", "emblem", "eye", "eyes", "face", "figure", "gem",
    "gemstone", "glass wall", "hand", "head", "helmet", "jeans", "logo", "mask", "mirror",
    "person", "railing", "ring", "shield", "sign", "silhouette", "skirt", "statue", "suit",
    "sword", "wall", "window",
}

STYLE_ADJECTIVE_MARKERS = {
    "abstract", "cartoonish", "colorful", "detailed", "dreamy", "foggy", "gothic",
    "minimal", "ornamental", "psychedelic", "soft", "surreal",
}

COMPOSITION_MARKERS = {
    "asymmetry", "centered", "central", "close-up", "dense", "framing", "horizontal", "landscape",
    "minimal", "negative space", "portrait", "symmetry", "symmetric", "vertical", "wide",
}


def merge_aesthetic_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge scraped aesthetics with curated seed data without duplicate names."""

    merged: dict[str, dict[str, Any]] = {}
    for item in [*AESTHETIC_SEED_DATA, *items]:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key not in merged:
            merged[key] = dict(item)
            merged[key].setdefault("source", item.get("source", "curated seed"))
            if item.get("source_url"):
                merged[key]["source_urls"] = [item.get("source_url")]
            continue
        target = merged[key]
        sources = {str(value).strip() for value in str(target.get("source", "")).split(" + ") if str(value).strip()}
        if item.get("source"):
            sources.add(str(item.get("source")))
        if sources:
            target["source"] = " + ".join(sorted(sources))
        source_urls = list(target.get("source_urls", []) or [])
        if target.get("source_url"):
            source_urls.append(target.get("source_url"))
        if item.get("source_url"):
            source_urls.append(item.get("source_url"))
        deduped_urls = []
        for url in source_urls:
            if url and url not in deduped_urls:
                deduped_urls.append(url)
        if deduped_urls:
            target["source_urls"] = deduped_urls[:6]
        for field in ("colors", "emotions", "symbols", "styles", "tags", "related"):
            values = list(target.get(field, []) or []) + list(item.get(field, []) or [])
            target[field] = sorted({str(value).strip() for value in values if str(value).strip()})[:32]
        if len(str(item.get("description", ""))) > len(str(target.get("description", ""))):
            target["description"] = item.get("description", "")
        for field in ("motifs", "values", "era", "media", "source_url"):
            if item.get(field) and not target.get(field):
                target[field] = item.get(field)
    return sorted(merged.values(), key=lambda item: str(item.get("name", "")).lower())


def load_aesthetic_knowledge() -> list[dict[str, Any]]:
    """Load the local aesthetics database, always merged with seed data."""

    cached_items: list[dict[str, Any]] = []
    if AESTHETICS_CACHE_PATH.exists():
        try:
            payload = json.loads(AESTHETICS_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("aesthetics"), list):
                cached_items = payload["aesthetics"]
            if isinstance(payload, list):
                cached_items = payload
        except Exception as exc:
            print(f"[WARN] Cannot read aesthetics cache: {exc}", file=sys.stderr)
    return merge_aesthetic_sources(cached_items)


def scrape_aesthetic_sources(limit: int = 500, source: str = "all") -> dict[str, Any]:
    """Refresh the aesthetics cache from configured scraping sources."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    raw_cache = scrape_aesthetic_source_items(limit, source=source)
    aesthetics = list(raw_cache.get("aesthetics", []) or [])
    if len(aesthetics) < 5:
        aesthetics.extend({**item, "source": "local seed fallback"} for item in AESTHETIC_SEED_DATA)
    merged = merge_aesthetic_sources(aesthetics)
    cache = {
        **raw_cache,
        "created": time.time(),
        "total_after_merge": len(merged),
        "aesthetics": merged,
    }
    AESTHETICS_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def scrape_aesthetics_wiki(limit: int = 500) -> dict[str, Any]:
    """Compatibility helper for callers that request the wiki-only source."""

    return scrape_aesthetic_sources(limit, source="wiki")


def aesthetic_keywords(item: dict[str, Any]) -> set[str]:
    """Return searchable keywords from one aesthetic metadata record."""

    fields = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(str(value) for value in item.get("colors", []) if value),
        " ".join(str(value) for value in item.get("emotions", []) if value),
        " ".join(str(value) for value in item.get("symbols", []) if value),
        " ".join(str(value) for value in item.get("styles", []) if value),
        " ".join(str(value) for value in item.get("tags", []) if value),
        " ".join(str(value) for value in item.get("related", []) if value),
        str(item.get("motifs", "")),
        str(item.get("values", "")),
    ]
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", " ".join(fields).lower())
    stop = {
        "the", "and", "with", "for", "from", "that", "this", "les", "des", "une", "dans", "sur", "est", "are",
        "aesthetic", "aesthetics", "style", "visual", "image", "images", "color", "colors", "also", "can",
        "commonly", "community", "culture", "often", "based", "used", "wiki", "page", "may", "more",
    }
    return {word for word in words if len(word) > 2 and word not in stop}


def split_concept_values(value: Any) -> list[str]:
    """Split scraped list/string fields into clean concept labels."""

    if isinstance(value, list):
        raw_parts = [str(item) for item in value]
    else:
        raw_parts = re.split(r",|;|/|\n|\||·|•", str(value or ""))
    labels = []
    for part in raw_parts:
        clean = clean_concept_label(part)
        if clean:
            labels.append(clean)
    return labels


def extract_description_terms(text: str, max_terms: int = 18) -> list[str]:
    """Extract compact taxonomy terms from aesthetic names/descriptions."""

    lowered = re.sub(r"[^a-zÀ-ÿ0-9 -]+", " ", str(text or "").lower())
    tokens = [
        token
        for token in re.findall(r"[a-zÀ-ÿ][a-zÀ-ÿ0-9-]{2,}", lowered)
        if token not in WORLD_TEXT_STOPWORDS and not token.endswith(("ing", "ed"))
    ]
    counter: Counter[str] = Counter()
    for size in (2, 3):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            phrase = clean_concept_label(" ".join(tokens[idx : idx + size]), max_words=3)
            if phrase:
                counter[phrase] += size
    for token in tokens:
        clean = clean_concept_label(token, max_words=1)
        if clean:
            counter[clean] += 1
    return [term for term, _amount in counter.most_common(max_terms)]


def build_world_taxonomy(aesthetics: list[dict[str, Any]], max_per_type: int = 160) -> dict[str, list[str]]:
    """Build modality-specific label banks from aesthetic metadata."""

    counters: dict[str, Counter[str]] = {
        "aesthetic": Counter(),
        "symbol": Counter(),
        "style": Counter(),
        "tag": Counter(),
        "texture": Counter(),
        "object": Counter(),
        "composition": Counter(),
    }
    for item in aesthetics:
        name = clean_concept_label(str(item.get("name", "")), max_words=5)
        if name:
            counters["aesthetic"][name] += 6
        field_plan = {
            "symbols": ("symbol", 5),
            "styles": ("style", 5),
            "tags": ("tag", 3),
            "related": ("aesthetic", 2),
            "colors": ("tag", 1),
        }
        for field, (bucket, weight) in field_plan.items():
            for label in split_concept_values(item.get(field, [])):
                counters[bucket][label] += weight
                if any(marker in label for marker in TEXTURE_MARKERS):
                    counters["texture"][label] += weight
                if any(marker in label for marker in COMPOSITION_MARKERS):
                    counters["composition"][label] += weight
        for label in split_concept_values(item.get("motifs", "")):
            counters["symbol"][label] += 4
        for term in extract_description_terms(f"{item.get('name', '')} {item.get('description', '')}", 12):
            if any(marker in term for marker in TEXTURE_MARKERS):
                counters["texture"][term] += 3
            elif any(marker in term for marker in COMPOSITION_MARKERS):
                counters["composition"][term] += 3
            elif term.endswith(("core", "ism", "wave")):
                counters["style"][term] += 2
            else:
                counters["tag"][term] += 1

    taxonomy: dict[str, list[str]] = {}
    for bucket, counter in counters.items():
        labels = []
        seen: set[str] = set()
        for label, _amount in counter.most_common(max_per_type * 2):
            clean = clean_concept_label(label, max_words=5)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            labels.append(clean)
            if len(labels) >= max_per_type:
                break
        taxonomy[bucket] = labels
    return taxonomy


def modality_prompt(term: str, obs_type: str) -> str:
    """Convert a taxonomy label into a SigLIP text prompt."""

    label = clean_concept_label(term, max_words=6) or str(term).strip().lower()
    if obs_type == "aesthetic":
        return f"an image in the {label} aesthetic"
    if obs_type == "object":
        return f"an image containing {label} as a concrete visible object"
    if obs_type == "symbol":
        return f"an image using {label} as a symbolic motif or visual sign"
    if obs_type == "texture":
        return f"an image with {label} texture, material or surface quality"
    if obs_type == "style":
        return f"an image in a {label} visual style or medium"
    if obs_type == "composition":
        return f"an image with {label} composition or framing"
    return f"an image with {label} as a visible aesthetic concept"
