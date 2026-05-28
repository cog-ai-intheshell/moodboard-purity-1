"""Base contracts for AI model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelStatus:
    """Runtime status returned by every model adapter."""

    id: str
    available: bool
    active: bool
    details: dict[str, Any]


class ModelAdapter(Protocol):
    """Small protocol followed by loadable model wrappers."""

    model_id: str

    def status(self) -> ModelStatus:
        ...
