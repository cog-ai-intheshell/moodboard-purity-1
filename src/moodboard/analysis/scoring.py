"""Numerical scoring helpers used by analysis modules.

This module contains small, deterministic measurements and fallback features.
It intentionally avoids model loading so it can be reused by analysis, tests and
offline world-learning code without pulling the ML stack into memory.
"""

from __future__ import annotations

import colorsys
import math

from PIL import Image, ImageFilter, ImageStat

from src.moodboard.core.image_io import RESAMPLE
from src.moodboard.core.schemas import ImageInfo, clamp_float


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize a positive score dictionary into a simple distribution."""

    total = sum(max(0.0, value) for value in scores.values())
    if total <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: round(max(0.0, value) / total, 4) for key, value in scores.items()}


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity that tolerates empty or differently-sized vectors."""

    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[idx] * right[idx] for idx in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def edge_density(image: Image.Image) -> float:
    """Estimate visual density from edge energy on a small grayscale copy."""

    gray = image.copy().convert("L")
    gray.thumbnail((220, 220), RESAMPLE)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    gray.close()
    edges.close()
    return clamp_float((stat.mean[0] / 255.0) * 2.8, 0.0, 0.0, 1.0)


def hue_temperature(hue: float) -> float:
    """Return a rough warm/cool hue score used by heuristic fallback vectors."""

    warm_center = min(abs(hue - 0.06), abs(hue - 1.06))
    cool_center = abs(hue - 0.58)
    return clamp_float(0.5 + (cool_center - warm_center), 0.5, 0.0, 1.0)


def heuristic_vector(info: ImageInfo, palette: list[dict[str, object]], density: float) -> list[float]:
    """Build a compact non-ML image vector when embeddings are unavailable."""

    hue, saturation, value = info.hsv
    orientation_vec = {
        "landscape": [1.0, 0.0, 0.0],
        "portrait": [0.0, 1.0, 0.0],
        "square": [0.0, 0.0, 1.0],
    }.get(info.orientation, [0.0, 0.0, 1.0])
    warm = hue_temperature(hue)
    palette_values: list[float] = []
    for item in palette[:3]:
        red, green, blue = item["rgb"]  # type: ignore[index]
        ph, ps, pv = colorsys.rgb_to_hsv(float(red) / 255.0, float(green) / 255.0, float(blue) / 255.0)
        palette_values.extend([ph, ps, pv])
    while len(palette_values) < 9:
        palette_values.append(0.0)
    return [
        hue,
        saturation,
        value,
        info.brightness,
        info.contrast,
        density,
        warm,
        math.log(info.area + 1.0) / 20.0,
        *orientation_vec,
        *palette_values,
    ]


def infer_image_tags(info: ImageInfo, palette: list[dict[str, object]], density: float) -> dict[str, list[str]]:
    """Produce conservative visual fallback tags before model enrichment."""

    hue, saturation, value = info.hsv
    tags: set[str] = set()
    emotions: set[str] = set()
    styles: set[str] = set()
    symbols: set[str] = set()
    composition: set[str] = set()

    palette_names = {str(item["name"]).lower() for item in palette}
    if any("pink" in name or "lavender" in name or "foggy" in name for name in palette_names):
        tags.update({"pastel", "soft", "dreamy"})
        styles.update({"soft palette"})
    if any("purple" in name or "violet" in name for name in palette_names):
        tags.update({"mystic", "cosmic"})
        styles.update({"cool spectral palette"})
    if any("gold" in name or "ivory" in name or "white" in name for name in palette_names):
        tags.update({"luminous", "sacred"})
        styles.update({"high key palette"})
    if any("black" in name or "burgundy" in name for name in palette_names) or value < 0.30:
        tags.update({"dark", "gothic"})
        styles.update({"low key palette"})
    if any("green" in name or "sage" in name or "emerald" in name for name in palette_names):
        tags.update({"organic", "natural"})
        styles.update({"earth palette"})
    if saturation > 0.55 and info.contrast > 0.35:
        tags.update({"vivid", "energetic"})
        styles.update({"high contrast"})
    if info.brightness > 0.72:
        tags.update({"airy", "light"})
        styles.update({"soft focus"})
    if info.brightness < 0.35:
        tags.update({"shadowed", "nocturnal"})
    if density > 0.48:
        styles.update({"detailed", "ink", "ornamental"})
        composition.update({"dense"})
    else:
        styles.update({"minimal", "soft focus"})
        composition.update({"negative space"})
    if info.contrast > 0.45:
        styles.add("high contrast")
        composition.add("dramatic contrast")
    if info.orientation == "portrait":
        composition.add("verticality")
    elif info.orientation == "landscape":
        composition.add("horizontal flow")
    else:
        composition.add("central balance")
    if hue < 0.06 or hue > 0.94:
        tags.add("red dominant" if saturation > 0.45 else "warm muted")

    if not styles:
        styles.add("digital painting")
    if not tags:
        tags.add("balanced")

    return {
        "tags": sorted(tags),
        "emotions": sorted(emotions),
        "symbols": sorted(symbols),
        "styles": sorted(styles),
        "composition": sorted(composition),
    }
