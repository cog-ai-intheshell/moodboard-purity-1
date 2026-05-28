"""Bento page rendering helpers.

`layout.py` decides where images should go. This module turns those normalized
rectangles into actual PIL pages and keeps rendering independent from the HTTP
server and export layer.
"""

from __future__ import annotations

import math
import random
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.moodboard.bento.layout import (
    BENTO_GRIDS,
    GRID_SLOT_COUNT,
    assign_images_to_slots,
    build_auto_norm_rects,
    choose_best_grid_for_batch,
    crop_grid_for_images,
    custom_grid_to_bento_grid,
)
from src.moodboard.bento.optimizer import optimize_bento_plan
from src.moodboard.bento.pdf_export import page_to_png_bytes
from src.moodboard.core.image_io import analyze_images, open_asset_rgb, sort_infos
from src.moodboard.core.schemas import ImageInfo, UploadedImage

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow fallback
    RESAMPLE = Image.LANCZOS


def page_to_png(page: Image.Image, preview: bool = False) -> bytes:
    """Compatibility helper used by older callers."""

    return page_to_png_bytes(page, preview)


def constrain_spacing(page_size: tuple[int, int], margin: int, gap: int) -> tuple[float, float]:
    """Clamp margin and gap so tiny/custom pages still render."""

    page_width, page_height = page_size
    max_margin = max(0, min(page_width, page_height) / 2.0 - 2.0)
    margin = min(float(margin), max_margin)
    gap = min(float(gap), max(0.0, min(page_width, page_height) / 3.0))
    return margin, gap


def paste_image(
    page: Image.Image,
    source: Image.Image,
    box: tuple[float, float, float, float],
    fit_mode: str,
    border_radius: int,
) -> None:
    """Paste one source image into a page box with cover/contain behavior."""

    x0, y0, x1, y1 = box
    width = int(round(x1 - x0))
    height = int(round(y1 - y0))
    if width <= 0 or height <= 0:
        return

    if fit_mode == "contain":
        tile = source.copy()
        tile.thumbnail((width, height), RESAMPLE)
    else:
        tile = ImageOps.fit(source, (width, height), method=RESAMPLE, centering=(0.5, 0.5))

    paste_x = int(round(x0 + (width - tile.width) / 2.0))
    paste_y = int(round(y0 + (height - tile.height) / 2.0))

    if border_radius > 0:
        radius = min(int(border_radius), tile.width // 2, tile.height // 2)
        mask = Image.new("L", tile.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, tile.width - 1, tile.height - 1), radius=radius, fill=255)
        page.paste(tile, (paste_x, paste_y), mask)
    else:
        page.paste(tile, (paste_x, paste_y))


def make_page_from_norm_rects(
    infos: list[ImageInfo],
    rects_norm: list[tuple[float, float, float, float]],
    params: dict[str, Any],
) -> Image.Image:
    """Render a page from normalized Bento rectangles."""

    page_width, page_height = params["page_size"]
    margin, gap = constrain_spacing(params["page_size"], params["margin"], params["gap"])
    page = Image.new("RGB", (page_width, page_height), params["background_color"])

    outer_inset = max(0.0, margin - gap / 2.0)
    ox0, oy0 = outer_inset, outer_inset
    ox1, oy1 = page_width - outer_inset, page_height - outer_inset
    outer_width, outer_height = ox1 - ox0, oy1 - oy0
    inner = gap / 2.0

    for info, (xn0, yn0, xn1, yn1) in zip(infos, rects_norm):
        x0 = ox0 + xn0 * outer_width + inner
        y0 = oy0 + yn0 * outer_height + inner
        x1 = ox0 + xn1 * outer_width - inner
        y1 = oy0 + yn1 * outer_height - inner
        try:
            image = open_asset_rgb(info.asset)
            paste_image(page, image, (x0, y0, x1, y1), params["fit_mode"], params["border_radius"])
            image.close()
        except Exception as exc:
            print(f"[WARN] Cannot render {info.asset.filename}: {exc}", file=sys.stderr)

    return page


def make_simple_grid_page(infos: list[ImageInfo], params: dict[str, Any]) -> Image.Image:
    """Render the simple equal-cell grid mode."""

    page_width, page_height = params["page_size"]
    margin, gap = constrain_spacing(params["page_size"], params["margin"], params["gap"])
    page = Image.new("RGB", (page_width, page_height), params["background_color"])
    count = len(infos)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))

    available_width = max(1.0, page_width - 2 * margin - (cols - 1) * gap)
    available_height = max(1.0, page_height - 2 * margin - (rows - 1) * gap)
    cell_width = available_width / cols
    cell_height = available_height / rows

    for idx, info in enumerate(infos):
        row = idx // cols
        col = idx % cols
        x0 = margin + col * (cell_width + gap)
        y0 = margin + row * (cell_height + gap)
        x1 = x0 + cell_width
        y1 = y0 + cell_height
        try:
            image = open_asset_rgb(info.asset)
            paste_image(page, image, (x0, y0, x1, y1), params["fit_mode"], params["border_radius"])
            image.close()
        except Exception as exc:
            print(f"[WARN] Cannot render {info.asset.filename}: {exc}", file=sys.stderr)

    return page


def render_pages(assets: list[UploadedImage], params: dict[str, Any]) -> tuple[list[Image.Image], list[ImageInfo], int]:
    """Analyze uploaded images, choose a layout, and render Bento pages."""

    infos = analyze_images(assets, params["orientation_threshold"])
    if not infos:
        raise ValueError("No usable images were found.")

    if params["use_color_gradient"] and not params.get("bento_optimizer"):
        infos = sort_infos(infos, params["gradient_mode"])

    layout_mode = params["layout_mode"]
    custom_grid = custom_grid_to_bento_grid(params["custom_grid"], params["orientation_threshold"]) if layout_mode == "custom" else None

    if params.get("bento_optimizer"):
        plan = optimize_bento_plan(infos, params, custom_grid)
        pages = [make_page_from_norm_rects(page_spec.infos, page_spec.rects, params) for page_spec in plan.pages]
        return pages, infos, max(1, plan.images_per_page)

    images_per_page = params["images_per_page"]
    if layout_mode in {"grid", "random"}:
        images_per_page = min(images_per_page, GRID_SLOT_COUNT)
    elif layout_mode == "custom" and custom_grid is not None:
        images_per_page = min(images_per_page, len(custom_grid["rects"]))
    images_per_page = max(1, images_per_page)

    pages: list[Image.Image] = []
    for start in range(0, len(infos), images_per_page):
        batch = infos[start : start + images_per_page]
        if layout_mode == "simple":
            pages.append(make_simple_grid_page(batch, params))
        elif layout_mode == "auto":
            rects = build_auto_norm_rects(len(batch))
            pages.append(make_page_from_norm_rects(batch, rects, params))
        elif layout_mode == "custom" and custom_grid is not None:
            grid = crop_grid_for_images(custom_grid, len(batch))
            ordered_infos = assign_images_to_slots(batch, grid)
            pages.append(make_page_from_norm_rects(ordered_infos, grid["rects"], params))
        else:
            source_grid = random.choice(BENTO_GRIDS) if layout_mode == "random" else choose_best_grid_for_batch(batch)
            grid = crop_grid_for_images(source_grid, len(batch))
            ordered_infos = assign_images_to_slots(batch, grid)
            pages.append(make_page_from_norm_rects(ordered_infos, grid["rects"], params))

    return pages, infos, images_per_page
