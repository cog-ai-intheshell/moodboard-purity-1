"""Aesthetics metadata API adapter."""

from __future__ import annotations

from typing import Any


def aesthetics_payload(*, refresh: bool = False, limit: int = 500, source: str = "all") -> dict[str, Any]:
    from src.moodboard.ai.variables.taxonomies import load_aesthetic_knowledge, scrape_aesthetic_sources

    if refresh:
        return scrape_aesthetic_sources(limit, source=source)
    return {"source": "cache-or-seed", "aesthetics": load_aesthetic_knowledge()}
