"""Model registry API adapter."""

from __future__ import annotations

from typing import Any


def model_status() -> dict[str, Any]:
    """Expose the model registry through the API layer."""

    from src.moodboard.ai.registry import runtime_registry

    return runtime_registry()
