"""Adapter for local image folders."""

from __future__ import annotations

from pathlib import Path

from src.moodboard.core.image_io import VALID_EXTENSIONS


def iter_image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS)
