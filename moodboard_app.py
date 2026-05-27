#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local Bento Moodboard web app.

This file fuses the three original scripts:
- simple A4 grid extraction from images_extractor.py
- organic recursive bento layout from bento_auto.py
- advanced grid/hero/color logic from bento_complex.py

Run:
    python3 moodboard_app.py

Then open:
    http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import colorsys
import io
import json
import math
import os
import random
import re
import sys
import time
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps, ImageStat


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "moodboard_interface.html"
PREVIEW_CACHE: dict[str, dict[str, Any]] = {}
PREVIEW_TTL_SECONDS = 20 * 60
PREVIEW_MAX_DIMENSION = 1400

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VALID_GRADIENT_MODES = {
    "default",
    "accent",
    "hue_buckets",
    "dark_to_light",
    "light_to_dark",
    "contrast",
    "hero",
}
VALID_LAYOUT_MODES = {"grid", "random", "custom", "auto", "simple"}
PAGE_PRESETS = {
    "a4_landscape": (3508, 2480),
    "a4_portrait": (2480, 3508),
    "screen_16_9": (4096, 2306),
    "iphone": (393, 852),
    "custom": (3508, 2480),
}

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow fallback
    RESAMPLE = Image.LANCZOS


@dataclass
class UploadedImage:
    filename: str
    data: bytes


@dataclass
class ImageInfo:
    asset: UploadedImage
    width: int
    height: int
    orientation: str
    area: int
    hsv: tuple[float, float, float]
    brightness: float
    contrast: float
    hero_score: float
    accent_h: float


BENTO_GRID_MATRICES = [
    {
        "name": "Grid 1 - hero 3x1 top",
        "matrix": [
            "A A A G H I",
            "B C D D E F",
            "B C D D E F",
            "J J K K L L",
        ],
    },
    {
        "name": "Grid 2 - hero 2x2 top-left",
        "matrix": [
            "A A D D E E",
            "A A H H F G",
            "B C I I F G",
            "B C J K L L",
        ],
    },
    {
        "name": "Grid 3 - variation",
        "matrix": [
            "A A B B C C",
            "D E B B F F",
            "D E G H I I",
            "J K G H L L",
        ],
    },
]


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def parse_color(value: Any, default: str = "#0f0f0f") -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(clamp_int(v, 0, 0, 255) for v in value)  # type: ignore[return-value]

    text = str(value or default).strip()
    if not text.startswith("#"):
        text = f"#{text}"
    if re.fullmatch(r"#[0-9a-fA-F]{3}", text):
        return tuple(int(ch * 2, 16) for ch in text[1:])  # type: ignore[return-value]
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return tuple(int(text[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    return parse_color(default)


def sanitize_filename(name: str, fallback: str = "moodboard") -> str:
    stem = os.path.basename(name or fallback)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem[:96] or fallback


def page_size_from_params(raw: dict[str, Any]) -> tuple[int, int]:
    preset = str(raw.get("pagePreset", "a4_landscape"))
    if preset != "custom":
        return PAGE_PRESETS.get(preset, PAGE_PRESETS["a4_landscape"])
    width = clamp_int(raw.get("pageWidth"), 3508, 320, 6000)
    height = clamp_int(raw.get("pageHeight"), 2480, 320, 6000)
    return width, height


def normalize_custom_grid(raw: dict[str, Any]) -> dict[str, Any]:
    source = raw.get("customGrid")
    if not isinstance(source, dict):
        source = {}

    cols = clamp_int(source.get("cols"), 6, 1, 12)
    rows = clamp_int(source.get("rows"), 4, 1, 12)
    raw_rects = source.get("rects")
    if not isinstance(raw_rects, list):
        raw_rects = []

    rects = []
    for item in raw_rects[:48]:
        if not isinstance(item, dict):
            continue
        x = clamp_int(item.get("x"), 0, 0, cols - 1)
        y = clamp_int(item.get("y"), 0, 0, rows - 1)
        width = clamp_int(item.get("w"), 1, 1, cols - x)
        height = clamp_int(item.get("h"), 1, 1, rows - y)
        rects.append({"x": x, "y": y, "w": width, "h": height})

    return {"cols": cols, "rows": rows, "rects": rects}


def normalize_params(raw: dict[str, Any]) -> dict[str, Any]:
    page_width, page_height = page_size_from_params(raw)
    layout_mode = str(raw.get("layoutMode", "grid"))
    if layout_mode not in VALID_LAYOUT_MODES:
        layout_mode = "grid"

    gradient_mode = str(raw.get("gradientMode", "accent"))
    if gradient_mode not in VALID_GRADIENT_MODES:
        gradient_mode = "accent"

    formats = raw.get("formats")
    if isinstance(formats, str):
        formats = [formats]
    if not isinstance(formats, list):
        formats = ["pdf"]
    formats = [str(fmt).lower() for fmt in formats if str(fmt).lower() in {"pdf", "png"}]
    if not formats:
        formats = ["pdf"]

    fit_mode = "cover" if parse_bool(raw.get("fillBlocks"), True) else "contain"

    return {
        "layout_mode": layout_mode,
        "images_per_page": clamp_int(raw.get("imagesPerPage"), 12, 1, 30),
        "auto_images_per_page": parse_bool(raw.get("autoImagesPerPage"), False),
        "background_color": parse_color(raw.get("backgroundColor"), "#0f0f0f"),
        "gap": clamp_int(raw.get("gap"), 30, 0, 180),
        "margin": clamp_int(raw.get("margin"), 30, 0, 260),
        "border_radius": clamp_int(raw.get("borderRadius"), 20, 0, 120),
        "use_color_gradient": parse_bool(raw.get("useColorGradient"), True),
        "gradient_mode": gradient_mode,
        "orientation_threshold": clamp_float(raw.get("orientationThreshold"), 1.2, 1.01, 2.5),
        "page_size": (page_width, page_height),
        "fit_mode": fit_mode,
        "formats": formats,
        "custom_grid": normalize_custom_grid(raw),
    }


def classify_orientation_from_ratio(ratio: float, threshold: float) -> str:
    if ratio > threshold:
        return "landscape"
    if ratio < 1.0 / threshold:
        return "portrait"
    return "square"


def classify_orientation(width: int, height: int, threshold: float) -> str:
    if height <= 0:
        return "square"
    return classify_orientation_from_ratio(width / float(height), threshold)


def build_bento_grids() -> tuple[list[dict[str, Any]], int]:
    grids: list[dict[str, Any]] = []
    expected_slots = None

    for grid_definition in BENTO_GRID_MATRICES:
        rows = [row.strip().split() for row in grid_definition["matrix"] if row.strip()]
        if not rows:
            raise ValueError(f"Empty grid: {grid_definition['name']}")

        row_count = len(rows)
        col_count = len(rows[0])
        for row in rows:
            if len(row) != col_count:
                raise ValueError(f"Rows have different lengths in {grid_definition['name']}")

        boxes: dict[str, list[int]] = {}
        for y, row in enumerate(rows):
            for x, letter in enumerate(row):
                if letter not in boxes:
                    boxes[letter] = [x, y, x + 1, y + 1]
                else:
                    box = boxes[letter]
                    box[0] = min(box[0], x)
                    box[1] = min(box[1], y)
                    box[2] = max(box[2], x + 1)
                    box[3] = max(box[3], y + 1)

        for letter, (x0, y0, x1, y1) in boxes.items():
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if rows[yy][xx] != letter:
                        raise ValueError(f"Grid letter {letter} is not rectangular.")

        rects = []
        areas = []
        orientations = []
        for letter in sorted(boxes.keys()):
            x0, y0, x1, y1 = boxes[letter]
            cell_w = x1 - x0
            cell_h = y1 - y0
            rects.append((x0 / col_count, y0 / row_count, x1 / col_count, y1 / row_count))
            areas.append(cell_w * cell_h)
            orientations.append(classify_orientation_from_ratio(cell_w / float(cell_h), 1.2))

        if expected_slots is None:
            expected_slots = len(rects)
        elif len(rects) != expected_slots:
            raise ValueError("All bento grids must expose the same slot count.")

        max_area = max(areas)
        grids.append(
            {
                "name": grid_definition["name"],
                "rects": rects,
                "slot_areas": areas,
                "slot_orientations": orientations,
                "hero_indices": [idx for idx, area in enumerate(areas) if area >= max_area * 0.9],
                "orientation_counts": Counter(orientations),
            }
        )

    return grids, int(expected_slots or 0)


BENTO_GRIDS, GRID_SLOT_COUNT = build_bento_grids()


def split_rect_equally(rect: tuple[float, float, float, float], count: int) -> list[tuple[float, float, float, float]]:
    x0, y0, x1, y1 = rect
    if count <= 0:
        return []
    if count == 1:
        return [rect]

    width = x1 - x0
    height = y1 - y0
    if width >= height:
        left_count = count // 2
        x_mid = x0 + width * (left_count / float(count))
        return split_rect_equally((x0, y0, x_mid, y1), left_count) + split_rect_equally(
            (x_mid, y0, x1, y1), count - left_count
        )

    top_count = count // 2
    y_mid = y0 + height * (top_count / float(count))
    return split_rect_equally((x0, y0, x1, y_mid), top_count) + split_rect_equally(
        (x0, y_mid, x1, y1), count - top_count
    )


def build_auto_norm_rects(count: int) -> list[tuple[float, float, float, float]]:
    rects = split_rect_equally((0.0, 0.0, 1.0, 1.0), count)
    return sorted(rects, key=lambda rc: ((rc[1] + rc[3]) / 2.0, (rc[0] + rc[2]) / 2.0))


def open_asset_rgb(asset: UploadedImage) -> Image.Image:
    with Image.open(io.BytesIO(asset.data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def analyze_images(assets: list[UploadedImage], orientation_threshold: float) -> list[ImageInfo]:
    infos: list[ImageInfo] = []
    accent_s_min = 0.3
    accent_v_min = 0.2

    for asset in assets:
        try:
            with Image.open(io.BytesIO(asset.data)) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                orientation = classify_orientation(width, height, orientation_threshold)

                image_small = image.copy()
                image_small.thumbnail((256, 256), RESAMPLE)

                rgb = image_small.convert("RGB")
                stat_rgb = ImageStat.Stat(rgb)
                red, green, blue = stat_rgb.mean
                hue, saturation, value = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)

                gray = image_small.convert("L")
                stat_gray = ImageStat.Stat(gray)
                brightness = stat_gray.mean[0] / 255.0
                contrast = stat_gray.stddev[0] / 128.0

                area = width * height
                hero_score = math.log(area + 1.0) / 10.0 + contrast * 1.5 + brightness

                hsv_image = image_small.convert("HSV")
                if hasattr(hsv_image, "get_flattened_data"):
                    hsv_pixels = list(hsv_image.get_flattened_data())
                else:
                    hsv_pixels = list(hsv_image.getdata())
                accent_candidates = [
                    pixel_h / 255.0
                    for pixel_h, pixel_s, pixel_v in hsv_pixels
                    if pixel_s / 255.0 >= accent_s_min and pixel_v / 255.0 >= accent_v_min
                ]
                accent_h = hue
                if accent_candidates:
                    sum_cos = 0.0
                    sum_sin = 0.0
                    for hue_norm in accent_candidates:
                        angle = 2.0 * math.pi * hue_norm
                        sum_cos += math.cos(angle)
                        sum_sin += math.sin(angle)
                    if sum_cos != 0.0 or sum_sin != 0.0:
                        accent_h = (math.atan2(sum_sin, sum_cos) / (2.0 * math.pi)) % 1.0

                infos.append(
                    ImageInfo(
                        asset=asset,
                        width=width,
                        height=height,
                        orientation=orientation,
                        area=area,
                        hsv=(hue, saturation, value),
                        brightness=brightness,
                        contrast=contrast,
                        hero_score=hero_score,
                        accent_h=accent_h,
                    )
                )
        except Exception as exc:
            print(f"[WARN] Cannot analyze {asset.filename}: {exc}", file=sys.stderr)

    return infos


def sort_infos(infos: list[ImageInfo], mode: str) -> list[ImageInfo]:
    def default_key(info: ImageInfo) -> tuple[float, float, float]:
        hue, saturation, value = info.hsv
        hue_sort = 2.0 if saturation < 0.15 else hue
        return hue_sort, value, saturation

    if mode == "default":
        return sorted(infos, key=default_key)
    if mode == "accent":
        return sorted(infos, key=lambda info: (info.accent_h, info.brightness))
    if mode == "dark_to_light":
        return sorted(infos, key=lambda info: info.brightness)
    if mode == "light_to_dark":
        return sorted(infos, key=lambda info: -info.brightness)
    if mode == "contrast":
        return sorted(infos, key=lambda info: (-info.contrast, info.brightness))
    if mode == "hero":
        return sorted(infos, key=lambda info: -info.hero_score)
    if mode == "hue_buckets":
        dark: list[ImageInfo] = []
        colored: list[ImageInfo] = []
        low_sat: list[ImageInfo] = []
        for info in infos:
            hue, saturation, value = info.hsv
            if value < 0.35:
                dark.append(info)
            elif saturation < 0.25:
                low_sat.append(info)
            else:
                colored.append(info)
        dark.sort(key=lambda info: info.brightness)
        colored.sort(key=lambda info: (info.hsv[0], info.hsv[2], info.hsv[1]))
        low_sat.sort(key=lambda info: info.brightness)
        return dark + colored + low_sat

    return sorted(infos, key=default_key)


def auto_images_per_page(total_images: int) -> int:
    if total_images <= 0:
        return 0
    guess = int(math.sqrt(total_images) * 3)
    return max(6, min(20, guess))


def choose_best_grid_for_batch(batch_infos: list[ImageInfo]) -> dict[str, Any]:
    image_counts = Counter(info.orientation for info in batch_infos)
    best_score = None
    best_grid = BENTO_GRIDS[0]

    for grid in BENTO_GRIDS:
        grid_counts = grid["orientation_counts"]
        score = sum(abs(grid_counts.get(ori, 0) - image_counts.get(ori, 0)) for ori in ("landscape", "portrait", "square"))
        if best_score is None or score < best_score:
            best_score = score
            best_grid = grid

    return best_grid


def crop_grid_for_images(grid: dict[str, Any], slot_count: int) -> dict[str, Any]:
    indices = list(range(len(grid["rects"])))[:slot_count]
    index_map = {old: new for new, old in enumerate(indices)}
    rects = [grid["rects"][idx] for idx in indices]
    if rects and slot_count < len(grid["rects"]):
        min_x = min(rect[0] for rect in rects)
        min_y = min(rect[1] for rect in rects)
        max_x = max(rect[2] for rect in rects)
        max_y = max(rect[3] for rect in rects)
        width = max(0.001, max_x - min_x)
        height = max(0.001, max_y - min_y)
        rects = [
            (
                (rect[0] - min_x) / width,
                (rect[1] - min_y) / height,
                (rect[2] - min_x) / width,
                (rect[3] - min_y) / height,
            )
            for rect in rects
        ]
    slot_orientations = [grid["slot_orientations"][idx] for idx in indices]
    return {
        "name": grid["name"],
        "rects": rects,
        "slot_orientations": slot_orientations,
        "slot_areas": [grid["slot_areas"][idx] for idx in indices],
        "hero_indices": [index_map[idx] for idx in grid["hero_indices"] if idx in index_map],
        "orientation_counts": Counter(slot_orientations),
    }


def custom_grid_to_bento_grid(custom_grid: dict[str, Any], orientation_threshold: float) -> dict[str, Any]:
    cols = clamp_int(custom_grid.get("cols"), 6, 1, 12)
    rows = clamp_int(custom_grid.get("rows"), 4, 1, 12)
    raw_rects = custom_grid.get("rects")
    if not isinstance(raw_rects, list):
        raw_rects = []

    occupied: set[tuple[int, int]] = set()
    rects_cells: list[tuple[int, int, int, int]] = []

    for item in raw_rects:
        if not isinstance(item, dict):
            continue
        x = clamp_int(item.get("x"), 0, 0, cols - 1)
        y = clamp_int(item.get("y"), 0, 0, rows - 1)
        width = clamp_int(item.get("w"), 1, 1, cols - x)
        height = clamp_int(item.get("h"), 1, 1, rows - y)
        cells = {(xx, yy) for yy in range(y, y + height) for xx in range(x, x + width)}
        if cells & occupied:
            continue
        occupied |= cells
        rects_cells.append((x, y, width, height))

    for y in range(rows):
        for x in range(cols):
            if (x, y) not in occupied:
                rects_cells.append((x, y, 1, 1))

    rects_cells.sort(key=lambda rect: (rect[1], rect[0], -rect[3] * rect[2]))
    rects = []
    areas = []
    orientations = []
    for x, y, width, height in rects_cells:
        rects.append((x / cols, y / rows, (x + width) / cols, (y + height) / rows))
        area = width * height
        areas.append(area)
        orientations.append(classify_orientation_from_ratio(width / float(height), orientation_threshold))

    max_area = max(areas) if areas else 1
    return {
        "name": "Custom grid",
        "rects": rects,
        "slot_areas": areas,
        "slot_orientations": orientations,
        "hero_indices": [idx for idx, area in enumerate(areas) if area >= max_area * 0.9],
        "orientation_counts": Counter(orientations),
    }


def assign_images_to_slots(batch_infos: list[ImageInfo], grid: dict[str, Any]) -> list[ImageInfo]:
    slot_orientations = grid["slot_orientations"]
    hero_indices = grid["hero_indices"]
    remaining = list(batch_infos)
    slot_infos: list[ImageInfo | None] = [None] * len(grid["rects"])

    for slot_idx in hero_indices:
        if not remaining:
            break
        slot_orientation = slot_orientations[slot_idx]
        best_idx = None
        best_info = None
        for idx, info in enumerate(remaining):
            if info.orientation == slot_orientation and (best_info is None or info.hero_score > best_info.hero_score):
                best_idx = idx
                best_info = info
        if best_idx is None:
            best_idx, best_info = max(enumerate(remaining), key=lambda item: item[1].hero_score)
        slot_infos[slot_idx] = best_info
        del remaining[best_idx]

    for slot_idx in range(len(slot_infos)):
        if slot_infos[slot_idx] is not None or not remaining:
            continue
        slot_orientation = slot_orientations[slot_idx]
        candidate_idx = next((idx for idx, info in enumerate(remaining) if info.orientation == slot_orientation), 0)
        slot_infos[slot_idx] = remaining[candidate_idx]
        del remaining[candidate_idx]

    fallback = batch_infos[0]
    return [info or fallback for info in slot_infos]


def constrain_spacing(page_size: tuple[int, int], margin: int, gap: int) -> tuple[float, float]:
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


def make_page_from_norm_rects(infos: list[ImageInfo], rects_norm: list[tuple[float, float, float, float]], params: dict[str, Any]) -> Image.Image:
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
    infos = analyze_images(assets, params["orientation_threshold"])
    if not infos:
        raise ValueError("No usable images were found.")

    if params["use_color_gradient"]:
        infos = sort_infos(infos, params["gradient_mode"])

    layout_mode = params["layout_mode"]
    custom_grid = custom_grid_to_bento_grid(params["custom_grid"], params["orientation_threshold"]) if layout_mode == "custom" else None

    images_per_page = auto_images_per_page(len(infos)) if params["auto_images_per_page"] else params["images_per_page"]
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


def pages_to_pdf_bytes(pages: list[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    first, rest = pages[0], pages[1:]
    first.save(buffer, "PDF", resolution=300.0, save_all=True, append_images=rest)
    return buffer.getvalue()


def page_to_png_bytes(page: Image.Image, preview: bool = False) -> bytes:
    output = page
    if preview:
        output = page.copy()
        output.thumbnail((1600, 1600), RESAMPLE)
    buffer = io.BytesIO()
    output.save(buffer, "PNG", dpi=(300, 300))
    if output is not page:
        output.close()
    return buffer.getvalue()


def page_to_preview_bytes(page: Image.Image) -> bytes:
    buffer = io.BytesIO()
    page.save(buffer, "PNG", compress_level=1)
    return buffer.getvalue()


def preview_params(params: dict[str, Any]) -> dict[str, Any]:
    page_width, page_height = params["page_size"]
    longest_side = max(page_width, page_height)
    if longest_side <= PREVIEW_MAX_DIMENSION:
        return dict(params)

    scale = PREVIEW_MAX_DIMENSION / float(longest_side)
    preview = dict(params)
    preview["page_size"] = (
        max(1, int(round(page_width * scale))),
        max(1, int(round(page_height * scale))),
    )
    preview["gap"] = max(0, int(round(params["gap"] * scale)))
    preview["margin"] = max(0, int(round(params["margin"] * scale)))
    preview["border_radius"] = max(0, int(round(params["border_radius"] * scale)))
    return preview


def build_export(pages: list[Image.Image], formats: list[str]) -> tuple[bytes, str, str]:
    wants_pdf = "pdf" in formats
    wants_png = "png" in formats

    if wants_pdf and not wants_png:
        return pages_to_pdf_bytes(pages), "application/pdf", "moodboard_bento.pdf"

    if wants_png and not wants_pdf and len(pages) == 1:
        return page_to_png_bytes(pages[0]), "image/png", "moodboard_bento_page_001.png"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if wants_pdf:
            archive.writestr("moodboard_bento.pdf", pages_to_pdf_bytes(pages))
        if wants_png:
            for idx, page in enumerate(pages, 1):
                archive.writestr(f"moodboard_bento_page_{idx:03d}.png", page_to_png_bytes(page))
    return buffer.getvalue(), "application/zip", "moodboard_bento_export.zip"


def cleanup_preview_cache() -> None:
    now = time.time()
    stale_ids = [
        preview_id
        for preview_id, item in PREVIEW_CACHE.items()
        if now - float(item.get("created", 0)) > PREVIEW_TTL_SECONDS
    ]
    for preview_id in stale_ids:
        PREVIEW_CACHE.pop(preview_id, None)


def parse_multipart_form(content_type: str, body: bytes) -> tuple[dict[str, str], list[UploadedImage]]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    fields: dict[str, str] = {}
    files: list[UploadedImage] = []

    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data.")

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            extension = os.path.splitext(filename.lower())[1]
            if extension in VALID_EXTENSIONS and payload:
                files.append(UploadedImage(sanitize_filename(filename, "image"), payload))
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="replace")

    return fields, files


def parse_request_payload(handler: BaseHTTPRequestHandler) -> tuple[dict[str, Any], list[UploadedImage]]:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0") or "0")
    body = handler.rfile.read(length)
    fields, files = parse_multipart_form(content_type, body)

    params: dict[str, Any] = {}
    if "params" in fields:
        try:
            decoded = json.loads(fields["params"])
            if isinstance(decoded, dict):
                params.update(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid params JSON: {exc}") from exc

    return normalize_params(params), files


class MoodboardHandler(BaseHTTPRequestHandler):
    server_version = "MoodboardApp/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def send_bytes(self, data: bytes, content_type: str, filename: str | None = None, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            safe_name = sanitize_filename(filename)
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path in {"/", "/index.html", "/moodboard_interface.html"}:
            if not HTML_PATH.exists():
                self.send_json({"error": "moodboard_interface.html is missing."}, status=500)
                return
            self.send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/api/health":
            self.send_json({"ok": True})
            return
        match = re.fullmatch(r"/api/preview/([A-Za-z0-9_-]+)/page-(\d+)\.png", self.path)
        if match:
            cleanup_preview_cache()
            preview_id = match.group(1)
            page_index = int(match.group(2)) - 1
            item = PREVIEW_CACHE.get(preview_id)
            pages = item.get("pages", []) if item else []
            if 0 <= page_index < len(pages):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(pages[page_index])))
                self.end_headers()
                self.wfile.write(pages[page_index])
                return
            self.send_json({"error": "Preview page not found."}, status=404)
            return
        self.send_json({"error": "Not found."}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        try:
            params, files = parse_request_payload(self)
            if not files:
                raise ValueError("Upload at least one image.")

            if self.path == "/api/preview":
                pages, infos, images_per_page = render_pages(files, preview_params(params))
                cleanup_preview_cache()
                preview_id = uuid.uuid4().hex
                preview_pages = [page_to_preview_bytes(page) for page in pages]
                page_count = len(pages)
                for page in pages:
                    page.close()
                PREVIEW_CACHE[preview_id] = {"created": time.time(), "pages": preview_pages}
                self.send_json(
                    {
                        "pages": [
                            f"/api/preview/{preview_id}/page-{idx:03d}.png"
                            for idx in range(1, page_count + 1)
                        ],
                        "imageCount": len(infos),
                        "pageCount": page_count,
                        "imagesPerPage": images_per_page,
                    }
                )
                return

            if self.path == "/api/generate":
                pages, infos, images_per_page = render_pages(files, params)
                data, content_type, filename = build_export(pages, params["formats"])
                for page in pages:
                    page.close()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("X-Image-Count", str(len(infos)))
                self.send_header("X-Page-Count", str(len(pages)))
                self.send_header("X-Images-Per-Page", str(images_per_page))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_json({"error": "Not found."}, status=404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self.send_json({"error": str(exc)}, status=400)
            except (BrokenPipeError, ConnectionResetError):
                return


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), MoodboardHandler)
    url = f"http://{host}:{port}"
    print(f"Moodboard Bento app running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Bento Moodboard web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind.")
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
