"""PDF, PNG and ZIP export helpers for Bento moodboards.

The image layout renderer is still being migrated out of the legacy app module.
This file owns serialization and report pages so export behavior has one stable
home during that migration.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.moodboard.core.cache import PREVIEW_MAX_DIMENSION

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow fallback
    RESAMPLE = Image.LANCZOS


def clamp_float(value: Any, default: float, low: float, high: float) -> float:
    """Parse and clamp report values before drawing score bars."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def pages_to_pdf_bytes(pages: list[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    first, rest = pages[0], pages[1:]
    first.save(buffer, "PDF", resolution=300.0, save_all=True, append_images=rest)
    return buffer.getvalue()


def pages_to_pdf(pages: list[Image.Image]) -> bytes:
    """Compatibility alias for callers that use the shorter facade name."""

    return pages_to_pdf_bytes(pages)


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
    page.save(buffer, "PNG", compress_level=0)
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


def load_report_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[float, float],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int,
) -> float:
    x, y = xy
    words = str(text).split()
    if not words:
        return y
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def draw_score_bar(
    draw: ImageDraw.ImageDraw,
    label: str,
    value: float,
    box: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    text: tuple[int, int, int],
    muted: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    bar_top = y0 + int((y1 - y0) * 0.52)
    draw.text((x0, y0), label, fill=text, font=font)
    draw.text((x1 - 92, y0), f"{round(value * 100)}%", fill=muted, font=font)
    draw.rounded_rectangle((x0, bar_top, x1, y1), radius=10, fill=(35, 35, 35))
    draw.rounded_rectangle((x0, bar_top, x0 + int((x1 - x0) * clamp_float(value, 0.0, 0.0, 1.0)), y1), radius=10, fill=accent)


def text_width(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), str(value), font=font)
    return int(bbox[2] - bbox[0])


def truncate_text(draw: ImageDraw.ImageDraw, value: Any, font: ImageFont.ImageFont, max_width: int) -> str:
    text = str(value or "")
    if text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    while text and text_width(draw, text + suffix, font) > max_width:
        text = text[:-1]
    return (text.rstrip() + suffix) if text else suffix


def rgb_from_hex(value: Any, default: tuple[int, int, int] = (93, 113, 252)) -> tuple[int, int, int]:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(char * 2 for char in raw)
    if len(raw) != 6:
        return default
    try:
        return tuple(int(raw[idx : idx + 2], 16) for idx in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return default


def color_to_rgb(color: dict[str, Any], default: tuple[int, int, int] = (80, 80, 80)) -> tuple[int, int, int]:
    rgb = color.get("rgb")
    if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
        try:
            return tuple(max(0, min(255, int(value))) for value in rgb[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    return rgb_from_hex(color.get("hex"), default)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    scale: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=int(14 * scale), fill=fill, outline=outline, width=max(1, int(scale)))


def percent_label(value: Any) -> str:
    return f"{round(clamp_float(value, 0.0, 0.0, 1.0) * 100)}%"


def graph_degrees(graph: dict[str, Any]) -> dict[str, float]:
    degrees: dict[str, float] = {}
    for edge in graph.get("edges", []) or []:
        weight = clamp_float(edge.get("weight", 0.0), 0.0, 0.0, 1.0)
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source:
            degrees[source] = degrees.get(source, 0.0) + weight
        if target:
            degrees[target] = degrees.get(target, 0.0) + weight
    return degrees


def collect_report_tags(analysis: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    graph = analysis.get("graph", {}) or {}
    degrees = graph_degrees(graph)
    bucket_names = {
        "aesthetic": "Aesthetics",
        "object": "Objects",
        "symbol": "Symbols",
        "style": "Styles",
        "texture": "Textures",
        "affect": "Affects",
        "emotion": "Emotions",
        "composition": "Composition",
        "color": "Colors",
    }
    order = ["Aesthetics", "Objects", "Symbols", "Affects", "Emotions", "Styles", "Textures", "Composition", "Colors"]
    buckets: dict[str, dict[str, tuple[str, float]]] = {name: {} for name in order}

    for node in graph.get("nodes", []) or []:
        node_type = str(node.get("type", ""))
        bucket = bucket_names.get(node_type)
        label = str(node.get("label", "")).strip()
        if not bucket or not label:
            continue
        key = label.lower()
        score = degrees.get(str(node.get("id", "")), 0.0) + clamp_float(node.get("weight", 0.0), 0.0, 0.0, 3.0)
        previous = buckets[bucket].get(key)
        if previous is None or score > previous[1]:
            buckets[bucket][key] = (label, score)

    for match in analysis.get("aestheticMatches", []) or []:
        label = str(match.get("name", "")).strip()
        if label:
            key = label.lower()
            score = clamp_float(match.get("score", 0.0), 0.0, 0.0, 1.0) * 3.0
            previous = buckets["Aesthetics"].get(key)
            if previous is None or score > previous[1]:
                buckets["Aesthetics"][key] = (label, score)

    for color in analysis.get("palette", []) or []:
        label = str(color.get("name", color.get("hex", ""))).strip()
        if label:
            key = label.lower()
            score = clamp_float(color.get("weight", 0.0), 0.0, 0.0, 1.0) * 3.0
            previous = buckets["Colors"].get(key)
            if previous is None or score > previous[1]:
                buckets["Colors"][key] = (label, score)

    return {
        name: sorted(values.values(), key=lambda item: item[1], reverse=True)
        for name, values in buckets.items()
        if values
    }


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: tuple[int, int, int],
    scale: float,
    fonts: dict[str, ImageFont.ImageFont],
    text: tuple[int, int, int],
    muted: tuple[int, int, int],
    panel: tuple[int, int, int],
    line: tuple[int, int, int],
    caption: str = "",
) -> None:
    x0, y0, x1, y1 = box
    draw_panel(draw, box, scale, panel, line)
    draw.rounded_rectangle((x0, y0, x1, y0 + int(7 * scale)), radius=int(8 * scale), fill=accent)
    draw.text((x0 + int(22 * scale), y0 + int(22 * scale)), label.upper(), fill=muted, font=fonts["small_bold"])
    draw.text((x0 + int(22 * scale), y0 + int(56 * scale)), value, fill=text, font=fonts["value"])
    if caption:
        draw.text((x0 + int(22 * scale), y1 - int(34 * scale)), truncate_text(draw, caption, fonts["small"], x1 - x0 - int(44 * scale)), fill=muted, font=fonts["small"])


def draw_chip(
    draw: ImageDraw.ImageDraw,
    label: str,
    xy: tuple[int, int],
    max_width: int,
    scale: float,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    text: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
) -> tuple[int, int]:
    x, y = xy
    pad_x = int(13 * scale)
    pad_y = int(8 * scale)
    height = int(34 * scale)
    label = truncate_text(draw, label, font, max(12, max_width - pad_x * 2))
    width = min(max_width, text_width(draw, label, font) + pad_x * 2)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=int(7 * scale), fill=fill, outline=outline)
    draw.text((x + pad_x, y + pad_y - int(1 * scale)), label, fill=text, font=font)
    return width, height


def draw_tag_buckets(
    draw: ImageDraw.ImageDraw,
    buckets: dict[str, list[tuple[str, float]]],
    box: tuple[int, int, int, int],
    scale: float,
    fonts: dict[str, ImageFont.ImageFont],
    text: tuple[int, int, int],
    muted: tuple[int, int, int],
    chip_fill: tuple[int, int, int],
    line: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    categories = list(buckets.items())[:8]
    if not categories:
        draw.text((x0, y0), "No extracted tags yet.", fill=muted, font=fonts["body"])
        return

    cols = 4 if (x1 - x0) > int(1600 * scale) else 2
    rows = (len(categories) + cols - 1) // cols
    gap = int(18 * scale)
    cell_w = (x1 - x0 - gap * (cols - 1)) // cols
    cell_h = max(int(126 * scale), (y1 - y0 - gap * (rows - 1)) // max(1, rows))

    for idx, (category, values) in enumerate(categories):
        col = idx % cols
        row = idx // cols
        cx = x0 + col * (cell_w + gap)
        cy = y0 + row * (cell_h + gap)
        draw.text((cx, cy), category.upper(), fill=muted, font=fonts["small_bold"])
        chip_x = cx
        chip_y = cy + int(32 * scale)
        max_y = cy + cell_h
        drawn = 0
        for label, _score in values:
            chip_w = min(cell_w, text_width(draw, label, fonts["small"]) + int(28 * scale))
            if chip_x + chip_w > cx + cell_w:
                chip_x = cx
                chip_y += int(42 * scale)
            if chip_y + int(34 * scale) > max_y:
                if chip_x + int(48 * scale) > cx + cell_w:
                    chip_x = cx
                    chip_y = max_y - int(34 * scale)
                draw_chip(draw, "...", (chip_x, chip_y), int(52 * scale), scale, fonts["small"], chip_fill, text, line)
                break
            width, _height = draw_chip(draw, label, (chip_x, chip_y), cell_w - (chip_x - cx), scale, fonts["small"], chip_fill, text, line)
            drawn += 1
            chip_x += width + int(8 * scale)
            if drawn >= 8 and len(values) > drawn:
                if chip_x + int(48 * scale) > cx + cell_w:
                    chip_x = cx
                    chip_y += int(42 * scale)
                if chip_y + int(34 * scale) <= max_y:
                    draw_chip(draw, "...", (chip_x, chip_y), int(52 * scale), scale, fonts["small"], chip_fill, text, line)
                break


def interpolate_rgb(colors: list[tuple[int, int, int]], amount: float) -> tuple[int, int, int]:
    if not colors:
        return (80, 80, 80)
    if len(colors) == 1:
        return colors[0]
    position = clamp_float(amount, 0.0, 0.0, 1.0) * (len(colors) - 1)
    left = int(position)
    right = min(len(colors) - 1, left + 1)
    mix = position - left
    return tuple(int(colors[left][idx] * (1.0 - mix) + colors[right][idx] * mix) for idx in range(3))  # type: ignore[return-value]


def draw_palette_panel(
    draw: ImageDraw.ImageDraw,
    palette: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    scale: float,
    fonts: dict[str, ImageFont.ImageFont],
    text: tuple[int, int, int],
    muted: tuple[int, int, int],
    panel: tuple[int, int, int],
    line: tuple[int, int, int],
) -> None:
    draw_panel(draw, box, scale, panel, line)
    x0, y0, x1, y1 = box
    pad = int(24 * scale)
    draw.text((x0 + pad, y0 + pad), "COLOR PALETTE", fill=muted, font=fonts["small_bold"])
    colors = [color_to_rgb(color) for color in palette[:10]]
    gradient_y = y0 + int(66 * scale)
    gradient_h = int(58 * scale)
    if colors:
        for x in range(x0 + pad, x1 - pad):
            amount = (x - x0 - pad) / max(1, (x1 - x0 - pad * 2))
            draw.line((x, gradient_y, x, gradient_y + gradient_h), fill=interpolate_rgb(colors, amount))
    else:
        draw.rounded_rectangle((x0 + pad, gradient_y, x1 - pad, gradient_y + gradient_h), radius=int(10 * scale), fill=(48, 48, 48))
    draw.rounded_rectangle((x0 + pad, gradient_y, x1 - pad, gradient_y + gradient_h), radius=int(10 * scale), outline=line, width=max(1, int(scale)))

    swatch_y = gradient_y + gradient_h + int(30 * scale)
    swatch_gap = int(14 * scale)
    swatch_count = min(6, len(palette))
    if swatch_count <= 0:
        draw.text((x0 + pad, swatch_y), "Palette not available.", fill=muted, font=fonts["body"])
        return
    swatch_w = (x1 - x0 - pad * 2 - swatch_gap * (swatch_count - 1)) // swatch_count
    swatch_h = int(58 * scale)
    for idx, color in enumerate(palette[:swatch_count]):
        sx = x0 + pad + idx * (swatch_w + swatch_gap)
        sy = swatch_y
        rgb = color_to_rgb(color)
        draw.rounded_rectangle((sx, sy, sx + swatch_w, sy + swatch_h), radius=int(8 * scale), fill=rgb)
        draw.text((sx, sy + swatch_h + int(10 * scale)), truncate_text(draw, color.get("name", color.get("hex", "")), fonts["small"], swatch_w), fill=text, font=fonts["small"])
        draw.text((sx, sy + swatch_h + int(34 * scale)), str(color.get("hex", "")).upper(), fill=muted, font=fonts["tiny"])


def node_positions(nodes: list[dict[str, Any]], box: tuple[int, int, int, int], scale: float) -> dict[str, tuple[float, float]]:
    x0, y0, x1, y1 = box
    pad = int(54 * scale)
    xs = [float(node.get("x", 0.0)) for node in nodes]
    ys = [float(node.get("y", 0.0)) for node in nodes]
    if not xs or not ys:
        return {}
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1e-9:
        max_x += 1.0
        min_x -= 1.0
    if abs(max_y - min_y) < 1e-9:
        max_y += 1.0
        min_y -= 1.0
    available_w = max(1, x1 - x0 - pad * 2)
    available_h = max(1, y1 - y0 - pad * 2)
    factor = min(available_w / (max_x - min_x), available_h / (max_y - min_y)) * 0.92
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    source_cx = (min_x + max_x) / 2.0
    source_cy = (min_y + max_y) / 2.0
    return {
        str(node.get("id")): (
            cx + (float(node.get("x", 0.0)) - source_cx) * factor,
            cy + (float(node.get("y", 0.0)) - source_cy) * factor,
        )
        for node in nodes
    }


def draw_point_cloud(
    draw: ImageDraw.ImageDraw,
    graph: dict[str, Any],
    box: tuple[int, int, int, int],
    scale: float,
    fonts: dict[str, ImageFont.ImageFont],
    text: tuple[int, int, int],
    muted: tuple[int, int, int],
    panel: tuple[int, int, int],
    line: tuple[int, int, int],
) -> None:
    draw_panel(draw, box, scale, panel, line)
    x0, y0, x1, y1 = box
    pad = int(28 * scale)
    draw.text((x0 + pad, y0 + pad), "LATENT POINT CLOUD", fill=muted, font=fonts["small_bold"])
    nodes = list(graph.get("nodes", []) or [])
    if not nodes:
        draw.text((x0 + pad, y0 + int(78 * scale)), "No graph nodes available.", fill=muted, font=fonts["body"])
        return

    plot_box = (x0 + pad, y0 + int(74 * scale), x1 - pad, y1 - int(74 * scale))
    positions = node_positions(nodes, plot_box, scale)
    degree = graph_degrees(graph)
    node_by_id = {str(node.get("id")): node for node in nodes}

    ring_color = (48, 48, 48)
    cx = (plot_box[0] + plot_box[2]) / 2
    cy = (plot_box[1] + plot_box[3]) / 2
    radius = min(plot_box[2] - plot_box[0], plot_box[3] - plot_box[1]) / 2
    for amount in (0.35, 0.62, 0.88):
        r = radius * amount
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ring_color, width=max(1, int(scale)))

    edges = sorted(graph.get("edges", []) or [], key=lambda edge: float(edge.get("weight", 0.0) or 0.0), reverse=True)[:260]
    for edge in edges:
        source = positions.get(str(edge.get("source", "")))
        target = positions.get(str(edge.get("target", "")))
        if not source or not target:
            continue
        weight = clamp_float(edge.get("weight", 0.0), 0.0, 0.0, 1.0)
        edge_value = int(74 + weight * 80)
        draw.line((source[0], source[1], target[0], target[1]), fill=(edge_value, edge_value, edge_value), width=max(1, int(scale * (0.55 + weight))))

    sorted_nodes = sorted(nodes, key=lambda node: degree.get(str(node.get("id", "")), 0.0), reverse=True)
    hub_ids = {str(node.get("id")) for node in sorted_nodes[:8]}
    type_boost = {"image": 1.2, "aesthetic": 1.45, "color": 0.75, "object": 0.95, "symbol": 1.0, "affect": 0.95, "emotion": 0.95, "style": 1.05, "texture": 0.8, "composition": 0.9}
    for node in sorted_nodes[::-1]:
        node_id = str(node.get("id"))
        pos = positions.get(node_id)
        if not pos:
            continue
        weight = clamp_float(node.get("weight", 0.2), 0.2, 0.0, 2.0)
        node_type = str(node.get("type", ""))
        radius_px = max(3, int((4.2 + weight * 7.5) * scale * type_boost.get(node_type, 1.0)))
        color = rgb_from_hex(node.get("clusterColor"), (93, 113, 252))
        if node_id in hub_ids:
            halo = radius_px + int(7 * scale)
            draw.ellipse((pos[0] - halo, pos[1] - halo, pos[0] + halo, pos[1] + halo), outline=(80, 80, 86), width=max(1, int(2 * scale)))
        draw.ellipse((pos[0] - radius_px, pos[1] - radius_px, pos[0] + radius_px, pos[1] + radius_px), fill=color, outline=(236, 236, 236), width=max(1, int(scale)))

    label_count = 0
    for node_id in hub_ids:
        node = node_by_id.get(node_id)
        pos = positions.get(node_id)
        if not node or not pos or str(node.get("type")) == "image":
            continue
        label = truncate_text(draw, node.get("label", ""), fonts["small"], int(230 * scale))
        draw.text((pos[0] + int(12 * scale), pos[1] - int(12 * scale)), label, fill=text, font=fonts["small"])
        label_count += 1
        if label_count >= 6:
            break

    cluster_counts: dict[str, int] = {}
    cluster_colors: dict[str, str] = {}
    for node in nodes:
        cluster = str(node.get("cluster", 0))
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        cluster_colors[cluster] = str(node.get("clusterColor", "#5D71FC"))
    legend_y = y1 - int(46 * scale)
    legend_x = x0 + pad
    for cluster in sorted(cluster_counts, key=lambda value: int(value) if value.isdigit() else value)[:8]:
        color = rgb_from_hex(cluster_colors.get(cluster))
        draw.ellipse((legend_x, legend_y, legend_x + int(14 * scale), legend_y + int(14 * scale)), fill=color)
        label = f"C{int(cluster) + 1 if cluster.isdigit() else cluster} {cluster_counts[cluster]}"
        draw.text((legend_x + int(20 * scale), legend_y - int(3 * scale)), label, fill=muted, font=fonts["tiny"])
        legend_x += int(92 * scale)


def draw_spectral_strip(
    draw: ImageDraw.ImageDraw,
    spectral: dict[str, Any],
    box: tuple[int, int, int, int],
    scale: float,
    fonts: dict[str, ImageFont.ImageFont],
    text: tuple[int, int, int],
    muted: tuple[int, int, int],
    panel: tuple[int, int, int],
    line: tuple[int, int, int],
    orange: tuple[int, int, int],
) -> None:
    draw_panel(draw, box, scale, panel, line)
    x0, y0, x1, y1 = box
    pad = int(24 * scale)
    draw.text((x0 + pad, y0 + pad), "SPECTRAL AESTHETIC ANALYSIS", fill=muted, font=fonts["small_bold"])
    values = spectral.get("eigenvalueEnergy") or spectral.get("energy") or []
    values = [clamp_float(value, 0.0, 0.0, 1.0) for value in values[:64]]
    chart = (x0 + pad, y0 + int(68 * scale), x1 - pad, y1 - int(76 * scale))
    draw.line((chart[0], chart[3], chart[2], chart[3]), fill=line, width=max(1, int(scale)))
    if values:
        max_value = max(values) or 1.0
        points: list[tuple[float, float]] = []
        for idx, value in enumerate(values):
            px = chart[0] + (chart[2] - chart[0]) * idx / max(1, len(values) - 1)
            py = chart[3] - (chart[3] - chart[1]) * (value / max_value)
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=orange, width=max(2, int(3 * scale)))
        for px, py in points[:18]:
            r = int(3 * scale)
            draw.ellipse((px - r, py - r, px + r, py + r), fill=orange)
    metrics = [
        ("Harmonicity", percent_label(spectral.get("harmonicityScore", 0.0))),
        ("Spectral purity", percent_label(spectral.get("spectralPurityScore", 0.0))),
        ("Regimes", str(spectral.get("aestheticRegimeCount", 0))),
        ("Gap", f"{clamp_float(spectral.get('spectralGap', 0.0), 0.0, 0.0, 99.0):.2f}"),
    ]
    metric_x = x0 + pad
    metric_y = y1 - int(48 * scale)
    for label, value in metrics:
        draw.text((metric_x, metric_y), f"{label}: ", fill=muted, font=fonts["tiny"])
        label_w = text_width(draw, f"{label}: ", fonts["tiny"])
        draw.text((metric_x + label_w, metric_y), value, fill=text, font=fonts["tiny_bold"])
        metric_x += int(190 * scale)


def make_analysis_pages(analysis: dict[str, Any], params: dict[str, Any]) -> list[Image.Image]:
    page_width, page_height = params["page_size"]
    bg = params["background_color"]
    text = (248, 248, 248)
    muted = (165, 170, 180)
    panel = (34, 34, 34)
    line = (58, 58, 58)
    blue = (93, 113, 252)
    orange = (248, 149, 64)
    red = (235, 87, 87)
    green = (39, 174, 96)

    scale = max(1.0, min(page_width, page_height) / 1800.0)
    margin = int(78 * scale)
    gap = int(30 * scale)
    fonts = {
        "title": load_report_font(int(42 * scale), bold=True),
        "heading": load_report_font(int(23 * scale), bold=True),
        "body": load_report_font(int(16 * scale)),
        "body_bold": load_report_font(int(16 * scale), bold=True),
        "small": load_report_font(int(12 * scale)),
        "small_bold": load_report_font(int(12 * scale), bold=True),
        "tiny": load_report_font(int(10 * scale)),
        "tiny_bold": load_report_font(int(10 * scale), bold=True),
        "value": load_report_font(int(34 * scale), bold=True),
    }

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        page = Image.new("RGB", (page_width, page_height), bg)
        draw = ImageDraw.Draw(page)
        step = max(42, int(54 * scale))
        for x in range(0, page_width, step):
            draw.line((x, 0, x, page_height), fill=(30, 30, 30), width=max(1, int(scale)))
        for y in range(0, page_height, step):
            draw.line((0, y, page_width, y), fill=(30, 30, 30), width=max(1, int(scale)))
        return page, draw

    pages: list[Image.Image] = []
    scores = analysis.get("scores", {})
    profile = analysis.get("globalProfile", {})
    dominant = profile.get("dominantAesthetic", {})
    palette = analysis.get("palette", [])
    clusters = analysis.get("clusters", [])
    outliers = analysis.get("outliers", [])
    matches = analysis.get("aestheticMatches", [])
    graph = analysis.get("graph", {}) or {}
    spectral = analysis.get("spectralAnalysis", {}) or {}
    tag_buckets = collect_report_tags(analysis)

    page, draw = new_page()
    draw.text((margin, margin), "Aesthetic Analysis Summary", fill=text, font=fonts["title"])
    y = margin + int(62 * scale)
    draw_wrapped_text(
        draw,
        f"Dominant aesthetic: {dominant.get('name', 'Unknown')} ({round(float(dominant.get('score', 0)) * 100)}%). "
        f"Mood: {profile.get('moodSummary', 'Balanced moodboard')}.",
        (margin, y),
        fonts["body"],
        muted,
        page_width - margin * 2,
        int(8 * scale),
    )
    y += int(98 * scale)
    card_w = (page_width - margin * 2 - gap * 3) // 4
    card_h = int(126 * scale)
    score_items = [
        ("Purity", percent_label(scores.get("purity", 0.0)), blue, "final moodboard score"),
        ("Harmony", percent_label(scores.get("harmonicity", scores.get("harmonyCoherence", 0.0))), orange, "spectral coherence"),
        ("Color", percent_label(scores.get("colorCoherence", 0.0)), green, "palette compactness"),
        ("Regimes", str(spectral.get("aestheticRegimeCount", len(clusters))), red, "latent modes"),
    ]
    for idx, (label, value, accent, caption) in enumerate(score_items):
        x = margin + idx * (card_w + gap)
        draw_metric_card(draw, (x, y, x + card_w, y + card_h), label, value, accent, scale, fonts, text, muted, panel, line, caption)

    y += card_h + gap
    mid_h = int(360 * scale)
    profile_w = int((page_width - margin * 2 - gap) * 0.48)
    profile_box = (margin, y, margin + profile_w, y + mid_h)
    palette_box = (profile_box[2] + gap, y, page_width - margin, y + mid_h)
    draw_panel(draw, profile_box, scale, panel, line)
    px = profile_box[0] + int(24 * scale)
    py = profile_box[1] + int(24 * scale)
    draw.text((px, py), "MOODBOARD PROFILE", fill=muted, font=fonts["small_bold"])
    py += int(44 * scale)
    dominant_label = str(dominant.get("name", "Unknown"))
    draw.text((px, py), truncate_text(draw, dominant_label, fonts["heading"], profile_box[2] - profile_box[0] - int(48 * scale)), fill=text, font=fonts["heading"])
    py += int(42 * scale)
    py = draw_wrapped_text(draw, str(profile.get("moodSummary", "Balanced moodboard")), (px, py), fonts["body"], muted, profile_box[2] - profile_box[0] - int(48 * scale), int(7 * scale))
    genome = profile.get("aestheticGenome", {}) or {}
    if genome:
        py += int(18 * scale)
        draw.text((px, py), "AESTHETIC GENOME", fill=muted, font=fonts["small_bold"])
        py += int(30 * scale)
        chip_x = px
        for term, amount in list(genome.items())[:9]:
            label = f"{str(term).title()} {round(float(amount) * 100)}%"
            width, height = draw_chip(draw, label, (chip_x, py), profile_box[2] - chip_x - int(24 * scale), scale, fonts["small"], (51, 51, 51), text, line)
            chip_x += width + int(8 * scale)
            if chip_x > profile_box[2] - int(150 * scale):
                chip_x = px
                py += height + int(8 * scale)
            if py > profile_box[3] - int(48 * scale):
                break

    draw_palette_panel(draw, palette, palette_box, scale, fonts, text, muted, panel, line)

    y += mid_h + gap
    tags_box = (margin, y, page_width - margin, page_height - margin)
    draw_panel(draw, tags_box, scale, panel, line)
    draw.text((tags_box[0] + int(24 * scale), tags_box[1] + int(24 * scale)), "EXTRACTED TAGS BY MODALITY", fill=muted, font=fonts["small_bold"])
    draw_tag_buckets(draw, tag_buckets, (tags_box[0] + int(24 * scale), tags_box[1] + int(68 * scale), tags_box[2] - int(24 * scale), tags_box[3] - int(24 * scale)), scale, fonts, text, muted, (55, 55, 55), line)
    pages.append(page)

    page, draw = new_page()
    draw.text((margin, margin), "Latent Aesthetic Map", fill=text, font=fonts["title"])
    y = margin + int(82 * scale)
    map_h = int((page_height - margin * 2) * 0.68)
    draw_point_cloud(draw, graph, (margin, y, page_width - margin, y + map_h), scale, fonts, text, muted, panel, line)
    y += map_h + gap
    draw_spectral_strip(draw, spectral, (margin, y, page_width - margin, page_height - margin), scale, fonts, text, muted, panel, line, orange)
    pages.append(page)

    page, draw = new_page()
    draw.text((margin, margin), "Clusters & Matches", fill=text, font=fonts["title"])
    y = margin + int(82 * scale)
    left_w = int((page_width - margin * 2 - gap) * 0.58)
    left_box = (margin, y, margin + left_w, page_height - margin)
    right_box = (left_box[2] + gap, y, page_width - margin, page_height - margin)
    draw_panel(draw, left_box, scale, panel, line)
    draw_panel(draw, right_box, scale, panel, line)

    cx = left_box[0] + int(24 * scale)
    cy = left_box[1] + int(24 * scale)
    draw.text((cx, cy), "CLUSTERS", fill=muted, font=fonts["small_bold"])
    cy += int(42 * scale)
    cluster_card_h = int(118 * scale)
    for cluster in clusters[:6]:
        if cy + cluster_card_h > left_box[3] - int(24 * scale):
            break
        cluster_color = rgb_from_hex(cluster.get("color"), blue)
        draw.rounded_rectangle((cx, cy, left_box[2] - int(24 * scale), cy + cluster_card_h), radius=int(11 * scale), fill=(39, 39, 39), outline=line)
        draw.ellipse((cx + int(18 * scale), cy + int(22 * scale), cx + int(42 * scale), cy + int(46 * scale)), fill=cluster_color)
        title = f"{cluster.get('label', 'Cluster')} - {cluster.get('size', 0)} images - {percent_label(cluster.get('share', 0.0))}"
        draw.text((cx + int(54 * scale), cy + int(17 * scale)), truncate_text(draw, title, fonts["body_bold"], left_box[2] - cx - int(92 * scale)), fill=text, font=fonts["body_bold"])
        concepts = [str(value).title() for value in cluster.get("concepts", [])[:7]]
        palette_names = [str(value).title() for value in cluster.get("palette", [])[:4]]
        draw.text((cx + int(54 * scale), cy + int(54 * scale)), truncate_text(draw, ", ".join(concepts) or "No dominant concepts", fonts["small"], left_box[2] - cx - int(92 * scale)), fill=muted, font=fonts["small"])
        if palette_names:
            draw.text((cx + int(54 * scale), cy + int(82 * scale)), truncate_text(draw, "Palette: " + ", ".join(palette_names), fonts["tiny"], left_box[2] - cx - int(92 * scale)), fill=muted, font=fonts["tiny"])
        cy += cluster_card_h + int(16 * scale)

    rx = right_box[0] + int(24 * scale)
    ry = right_box[1] + int(24 * scale)
    draw.text((rx, ry), "CLOSEST AESTHETICS", fill=muted, font=fonts["small_bold"])
    ry += int(40 * scale)
    for match in matches[:6]:
        value = clamp_float(match.get("score", 0.0), 0.0, 0.0, 1.0)
        draw_score_bar(draw, str(match.get("name", "Unknown")), value, (rx, ry, right_box[2] - int(24 * scale), ry + int(58 * scale)), blue if value >= 0.7 else orange, text, muted, fonts["small"])
        ry += int(78 * scale)

    ry += int(12 * scale)
    draw.text((rx, ry), "COHERENCE BREAKDOWN", fill=muted, font=fonts["small_bold"])
    ry += int(38 * scale)
    breakdown = [
        ("Hybridation", scores.get("hybridation", 0.0), orange),
        ("Style", scores.get("styleCoherence", 0.0), blue),
        ("Symbolic", scores.get("symbolicCoherence", 0.0), red),
        ("Emotion", scores.get("emotionalCoherence", 0.0), green),
        ("Composition", scores.get("compositionCoherence", 0.0), orange),
    ]
    for label, value, accent in breakdown:
        if ry + int(54 * scale) > right_box[3] - int(120 * scale):
            break
        draw_score_bar(draw, label, float(value), (rx, ry, right_box[2] - int(24 * scale), ry + int(48 * scale)), accent, text, muted, fonts["small"])
        ry += int(62 * scale)

    if outliers and ry < right_box[3] - int(86 * scale):
        ry += int(16 * scale)
        draw.text((rx, ry), "OUTLIERS", fill=red, font=fonts["small_bold"])
        ry += int(32 * scale)
        draw_wrapped_text(draw, ", ".join(item.get("filename", item.get("id", "")) for item in outliers[:8]), (rx, ry), fonts["small"], muted, right_box[2] - right_box[0] - int(48 * scale), int(6 * scale))
    pages.append(page)
    return pages


def build_export(
    pages: list[Image.Image],
    formats: list[str],
    analysis: dict[str, Any] | None = None,
    pdf_pages: list[Image.Image] | None = None,
) -> tuple[bytes, str, str]:
    wants_pdf = "pdf" in formats
    wants_png = "png" in formats
    pdf_source = pdf_pages or pages

    if wants_pdf and not wants_png:
        return pages_to_pdf_bytes(pdf_source), "application/pdf", "moodboard_bento.pdf"

    if wants_png and not wants_pdf and len(pages) == 1 and analysis is None:
        return page_to_png_bytes(pages[0]), "image/png", "moodboard_bento_page_001.png"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if wants_pdf:
            archive.writestr("moodboard_bento.pdf", pages_to_pdf_bytes(pdf_source))
        if wants_png:
            for idx, page in enumerate(pages, 1):
                archive.writestr(f"moodboard_bento_page_{idx:03d}.png", page_to_png_bytes(page))
        if analysis is not None:
            archive.writestr(
                "analysis.json",
                json.dumps(analysis, ensure_ascii=False, indent=2).encode("utf-8"),
            )
    return buffer.getvalue(), "application/zip", "moodboard_bento_export.zip"
