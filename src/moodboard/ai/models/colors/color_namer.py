"""Nearest color-name model backed by the local color database.

The application treats color naming as a small deterministic model: RGB input
is converted to CIE Lab, compared with the curated/scraped color database, then
lightness/saturation rules keep near-black, near-white and neutral colors sane.
"""

from __future__ import annotations

import colorsys
import json
import math

from src.moodboard.core.paths import COLOR_NAMES_PATH


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    """Return a browser-friendly hex value for an RGB color."""

    return "#{:02x}{:02x}{:02x}".format(*color)


COLOR_NAME_TABLE: list[tuple[str, tuple[int, int, int]]] = [
    ("Black", (8, 8, 8)),
    ("Charcoal", (34, 34, 34)),
    ("Graphite", (55, 57, 61)),
    ("Soft Gray", (142, 142, 142)),
    ("Silver Gray", (188, 190, 192)),
    ("Ivory White", (240, 235, 218)),
    ("Bone", (226, 218, 198)),
    ("Cream", (232, 214, 173)),
    ("Taupe", (129, 117, 101)),
    ("Espresso", (58, 39, 31)),
    ("Chocolate Brown", (96, 59, 38)),
    ("Burgundy", (92, 22, 42)),
    ("Crimson", (177, 32, 48)),
    ("Red", (214, 58, 55)),
    ("Coral", (229, 108, 88)),
    ("Dusty Pink", (198, 137, 158)),
    ("Pastel Pink", (231, 177, 203)),
    ("Magenta", (202, 59, 159)),
    ("Lavender", (174, 153, 216)),
    ("Violet", (129, 92, 203)),
    ("Deep Purple", (62, 42, 112)),
    ("Midnight Blue", (25, 38, 92)),
    ("Sky Blue", (102, 156, 214)),
    ("Foggy Blue", (148, 177, 194)),
    ("Cyan", (46, 177, 197)),
    ("Teal", (31, 125, 126)),
    ("Emerald", (42, 146, 89)),
    ("Sage Green", (135, 160, 124)),
    ("Olive", (107, 111, 55)),
    ("Lime", (155, 191, 55)),
    ("Gold", (204, 166, 72)),
    ("Ochre", (165, 122, 43)),
    ("Orange", (220, 116, 45)),
    ("Copper", (156, 84, 48)),
]
COLOR_NAME_TABLE_CACHE: list[tuple[str, tuple[int, int, int]]] | None = None


def srgb_channel_to_linear(value: float) -> float:
    """Convert an sRGB channel to linear RGB for perceptual color distance."""

    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def rgb_to_lab(color: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert RGB to CIE Lab using the D65 reference white."""

    red, green, blue = [srgb_channel_to_linear(channel / 255.0) for channel in color]
    x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041
    white = (0.95047, 1.0, 1.08883)

    def pivot(value: float) -> float:
        return value ** (1.0 / 3.0) if value > 0.008856 else 7.787 * value + 16.0 / 116.0

    fx, fy, fz = pivot(x / white[0]), pivot(y / white[1]), pivot(z / white[2])
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def lab_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    """Approximate perceptual distance between two RGB colors."""

    left_lab = rgb_to_lab(left)
    right_lab = rgb_to_lab(right)
    return math.sqrt(sum((left_lab[idx] - right_lab[idx]) ** 2 for idx in range(3)))


def load_color_name_table() -> list[tuple[str, tuple[int, int, int]]]:
    """Load scraped color names once, falling back to a compact built-in table."""

    global COLOR_NAME_TABLE_CACHE
    if COLOR_NAME_TABLE_CACHE is not None:
        return COLOR_NAME_TABLE_CACHE

    table = list(COLOR_NAME_TABLE)
    if COLOR_NAMES_PATH.exists():
        try:
            data = json.loads(COLOR_NAMES_PATH.read_text(encoding="utf-8"))
            scraped: list[tuple[str, tuple[int, int, int]]] = []
            for item in data.get("colors", []):
                item_name = str(item.get("name", "")).strip()
                rgb = item.get("rgb")
                if (
                    item_name
                    and isinstance(rgb, list)
                    and len(rgb) == 3
                    and all(isinstance(channel, int) and 0 <= channel <= 255 for channel in rgb)
                ):
                    scraped.append((item_name, (rgb[0], rgb[1], rgb[2])))
            if scraped:
                table = scraped + table
        except (OSError, ValueError, TypeError):
            table = list(COLOR_NAME_TABLE)

    deduped: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[tuple[str, tuple[int, int, int]]] = set()
    for item_name, rgb in table:
        key = (item_name.casefold(), rgb)
        if key not in seen:
            deduped.append((item_name, rgb))
            seen.add(key)
    COLOR_NAME_TABLE_CACHE = deduped
    return deduped


def color_name(color: tuple[int, int, int]) -> str:
    """Return the nearest useful color name for an RGB value."""

    hue, saturation, value = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    nearest_name, nearest_rgb = min(load_color_name_table(), key=lambda item: lab_distance(color, item[1]))
    if value < 0.055:
        return "Black"
    if saturation < 0.025 and value > 0.96:
        return "White"
    if saturation < 0.045 and value < 0.22:
        return "Charcoal"
    if saturation < 0.045 and 0.34 < value < 0.74 and lab_distance(color, nearest_rgb) > 13.0:
        return "Gray"
    if saturation < 0.12 and value > 0.72 and 0.06 <= hue <= 0.18 and lab_distance(color, nearest_rgb) > 16.0:
        return "Cream" if value > 0.82 else "Taupe"
    return nearest_name


def name(rgb: tuple[int, int, int]) -> str:
    """Adapter alias used by the modular AI registry."""

    return color_name(rgb)
