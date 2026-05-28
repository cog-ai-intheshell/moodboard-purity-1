"""AI orchestration boundary for moodboard analysis.

This module coordinates model adapters without owning the full analysis logic.
It keeps backend selection policies in one place so the analyzer can focus on
turning model outputs into scores, clusters and graph payloads.
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from src.moodboard.ai.models.captions.florence2 import try_florence_vision_tasks
from src.moodboard.ai.models.captions.smolvlm import try_fast_caption_model
from src.moodboard.core.schemas import VALID_ANALYSIS_DEPTHS


def run_vision_language_tasks(
    images: list[Image.Image],
    analysis_depth: str = "balanced",
) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]], str]:
    """Run the configured caption/region backend for a batch of images."""

    backend = os.environ.get("MOODBOARD_CAPTION_BACKEND", "florence").strip().lower()
    depth = analysis_depth if analysis_depth in VALID_ANALYSIS_DEPTHS else "balanced"
    if backend in {"0", "false", "no", "none", "disabled"}:
        return {}, {}, "disabled"
    if backend in {"fast", "smolvlm", "smolvlm2"}:
        captions, fast_status = try_fast_caption_model(images, depth)
        if captions:
            return captions, {}, fast_status
        if os.environ.get("MOODBOARD_FAST_FALLBACK_FLORENCE", "1").lower() in {"0", "false", "no"}:
            return {}, {}, fast_status
        florence_captions, regions, florence_status = try_florence_vision_tasks(images, depth)
        return florence_captions, regions, f"{fast_status}; fallback={florence_status}"
    if backend == "auto" and depth == "fast":
        captions, fast_status = try_fast_caption_model(images, depth)
        if captions:
            return captions, {}, fast_status
        florence_captions, regions, florence_status = try_florence_vision_tasks(images, depth)
        return florence_captions, regions, f"{fast_status}; fallback={florence_status}"
    if backend == "hybrid":
        captions, fast_status = try_fast_caption_model(images, depth)
        florence_captions, regions, florence_status = try_florence_vision_tasks(images, depth)
        merged = dict(florence_captions)
        merged.update(captions)
        return merged, regions, f"{fast_status}; regions={florence_status}"
    return try_florence_vision_tasks(images, depth)


def analyze(files: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    """Run the configured analysis pipeline for a moodboard."""

    from src.moodboard.analysis.moodboard_analyzer import analyze_moodboard

    return analyze_moodboard(files, params)
