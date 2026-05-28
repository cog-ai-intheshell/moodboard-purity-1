"""Generate/export API adapter used by the HTTP server."""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.moodboard.ai.orchestrator import analyze
from src.moodboard.core.cache import PREVIEW_CACHE, cleanup_preview_cache


def preview_payload(files: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    """Render low-resolution preview pages and store them in the preview cache."""

    from src.moodboard.bento.pdf_export import page_to_preview_bytes, preview_params
    from src.moodboard.bento.render import render_pages

    pages, infos, images_per_page = render_pages(files, preview_params(params))
    cleanup_preview_cache()
    preview_id = uuid.uuid4().hex
    preview_pages = [page_to_preview_bytes(page) for page in pages]
    page_count = len(pages)
    for page in pages:
        page.close()
    PREVIEW_CACHE[preview_id] = {"created": time.time(), "pages": preview_pages}
    return {
        "pages": [f"/api/preview/{preview_id}/page-{idx:03d}.png" for idx in range(1, page_count + 1)],
        "imageCount": len(infos),
        "pageCount": page_count,
        "imagesPerPage": images_per_page,
    }


def generate_export(files: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    """Render Bento pages, append PDF analysis pages and return export bytes."""

    from src.moodboard.bento.pdf_export import build_export, make_analysis_pages
    from src.moodboard.bento.render import render_pages

    pages, infos, images_per_page = render_pages(files, params)
    analysis = analyze(files, params)
    analysis_pages = make_analysis_pages(analysis, params) if "pdf" in params["formats"] else []
    pdf_pages = pages + analysis_pages if analysis_pages else pages
    data, content_type, filename = build_export(pages, params["formats"], analysis, pdf_pages)
    for page in pages:
        page.close()
    for page in analysis_pages:
        page.close()
    return {
        "data": data,
        "contentType": content_type,
        "filename": filename,
        "headers": {
            "X-Image-Count": str(len(infos)),
            "X-Page-Count": str(len(pages)),
            "X-Analysis-Page-Count": str(len(analysis_pages)),
            "X-Images-Per-Page": str(images_per_page),
        },
    }
