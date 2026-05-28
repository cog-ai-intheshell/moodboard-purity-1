"""Shared runtime caches.

The current HTTP server is still a lightweight stdlib server, so these caches
remain process-local. Keeping them in one module makes it easier to replace the
implementation later with a disk cache or a small key-value store.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any


PREVIEW_CACHE: dict[str, dict[str, Any]] = {}
PREVIEW_TTL_SECONDS = 20 * 60
PREVIEW_MAX_DIMENSION = 1400

ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
ANALYSIS_CACHE_TTL_SECONDS = 45 * 60

ML_MODEL_CACHE: dict[str, Any] = {}

# UMAP/Numba and the ML runtimes are not all safe under concurrent requests.
ANALYSIS_RUNTIME_LOCK = threading.RLock()


def cleanup_preview_cache() -> None:
    """Drop stale rendered preview PNGs from the process-local cache."""

    now = time.time()
    stale_ids = [
        preview_id
        for preview_id, item in PREVIEW_CACHE.items()
        if now - float(item.get("created", 0)) > PREVIEW_TTL_SECONDS
    ]
    for preview_id in stale_ids:
        PREVIEW_CACHE.pop(preview_id, None)


def cleanup_analysis_cache() -> None:
    """Drop stale analysis payloads from the process-local cache."""

    now = time.time()
    stale_keys = [
        key
        for key, item in ANALYSIS_CACHE.items()
        if now - float(item.get("created", 0)) > ANALYSIS_CACHE_TTL_SECONDS
    ]
    for key in stale_keys:
        ANALYSIS_CACHE.pop(key, None)


def analysis_cache_key(
    assets: list[Any],
    params: dict[str, Any],
    model_version: str,
    caption_backend: str | None = None,
) -> str:
    """Build a stable cache key from model version, params and uploaded bytes."""

    digest = hashlib.sha256()
    digest.update(model_version.encode("utf-8"))
    digest.update((caption_backend or os.environ.get("MOODBOARD_CAPTION_BACKEND", "florence")).encode("utf-8"))
    digest.update(json.dumps(params, sort_keys=True, default=str).encode("utf-8"))
    for asset in assets:
        digest.update(str(getattr(asset, "filename", "")).encode("utf-8", errors="ignore"))
        digest.update(hashlib.sha256(getattr(asset, "data", b"")).digest())
    return digest.hexdigest()
