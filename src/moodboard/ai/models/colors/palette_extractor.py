"""Local color palette extraction adapter.

This module is the color modality extractor used by the analyzer. It keeps all
palette logic outside the web server so the future AI orchestrator can swap it
for a learned palette model without touching request handling.
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image

from src.moodboard.ai.models.colors.color_namer import color_name, lab_distance, rgb_to_hex
from src.moodboard.core.schemas import clamp_float

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - Pillow < 9 compatibility.
    RESAMPLE = Image.LANCZOS


def image_palette(image: Image.Image, count: int = 12) -> list[dict[str, Any]]:
    """Extract the dominant image colors with stable names and roles."""

    sample = image.copy().convert("RGB")
    sample.thumbnail((260, 260), RESAMPLE)
    quantized = sample.quantize(colors=max(4, count * 4), method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    color_counts = quantized.getcolors(sample.width * sample.height) or []
    total = max(1, sum(item[0] for item in color_counts))
    ranked = sorted(color_counts, reverse=True)[: count * 2]

    out: list[dict[str, Any]] = []
    for amount, index in ranked:
        offset = index * 3
        if offset + 2 >= len(palette):
            continue
        rgb = tuple(int(v) for v in palette[offset : offset + 3])
        if any(lab_distance(rgb, tuple(existing["rgb"])) < 7.0 for existing in out):
            continue
        out.append(
            {
                "hex": rgb_to_hex(rgb),
                "rgb": list(rgb),
                "name": color_name(rgb),
                "weight": round(amount / total, 4),
                "role": "dominant" if not out else ("secondary" if len(out) < 4 else "accent"),
            }
        )
        if len(out) >= count:
            break
    sample.close()
    return out


def palette_coherence_score(palette: list[dict[str, Any]]) -> float:
    """Score whether the global palette forms a compact perceptual family."""

    usable = [
        item
        for item in palette
        if isinstance(item.get("rgb"), list)
        and len(item.get("rgb", [])) >= 3
        and float(item.get("weight", 0.0) or 0.0) > 0.0
    ]
    if not usable:
        return 0.0

    total_weight = max(1e-6, sum(float(item.get("weight", 0.0) or 0.0) for item in usable))
    weighted_rgb = tuple(
        int(
            round(
                sum(float(item.get("weight", 0.0) or 0.0) * float(item["rgb"][channel_idx]) for item in usable)
                / total_weight
            )
        )
        for channel_idx in range(3)
    )
    average_delta = sum(
        (float(item.get("weight", 0.0) or 0.0) / total_weight)
        * lab_distance(tuple(int(value) for value in item["rgb"][:3]), weighted_rgb)
        for item in usable
    )
    compactness = clamp_float(1.0 - (average_delta / 92.0), 0.0, 0.0, 1.0)
    top_weights = sorted((float(item.get("weight", 0.0) or 0.0) / total_weight for item in usable), reverse=True)
    concentration = clamp_float(sum(top_weights[: min(6, len(top_weights))]), 0.0, 0.0, 1.0)
    dominance = clamp_float(top_weights[0] if top_weights else 0.0, 0.0, 0.0, 1.0)
    long_tail_penalty = clamp_float((len(usable) - 12) / 72.0, 0.0, 0.0, 0.22)
    return clamp_float(
        (compactness * 0.58) + (concentration * 0.30) + (math.sqrt(dominance) * 0.12) - long_tail_penalty,
        0.0,
        0.0,
        1.0,
    )


def extract(image: Any, count: int = 12) -> list[dict[str, Any]]:
    """Registry-friendly alias for palette extraction."""

    return image_palette(image, count)
