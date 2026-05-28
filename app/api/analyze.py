"""Analyze API adapter.

The HTTP server enters the AI stack through `ai.orchestrator`, keeping endpoint
code independent from the concrete analyzer implementation.
"""

from __future__ import annotations

from typing import Any

from src.moodboard.ai.orchestrator import analyze


def analyze_images(files: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    """Run the public analysis entrypoint for uploaded images."""

    return analyze(files, params)
