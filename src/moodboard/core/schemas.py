"""Small shared data structures used across the moodboard pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any


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
VALID_ANALYSIS_DEPTHS = {"fast", "balanced", "deep"}
VALID_BENTO_OPTIMIZER_MODES = {"balanced", "editorial", "dense", "clustered"}
PAGE_PRESETS = {
    "a4_landscape": (3508, 2480),
    "a4_portrait": (2480, 3508),
    "screen_16_9": (4096, 2304),
    "iphone": (393, 852),
    "custom": (3508, 2480),
}


@dataclass
class UploadedImage:
    """Raw uploaded image payload before it is opened by PIL."""

    filename: str
    data: bytes


@dataclass
class ImageInfo:
    """Basic image measurements used by Bento layout and analysis."""

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


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    """Parse browser/config input into a bounded integer."""

    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def clamp_float(value: Any, default: float, low: float, high: float) -> float:
    """Parse browser/config input into a bounded float."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse checkbox/string form values into booleans."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def parse_color(value: Any, default: str = "#0f0f0f") -> tuple[int, int, int]:
    """Parse a hex or RGB-like value into an RGB tuple."""

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
    """Return a short filesystem/header-safe filename."""

    stem = os.path.basename(name or fallback)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem[:96] or fallback


def page_size_from_params(raw: dict[str, Any]) -> tuple[int, int]:
    """Resolve the selected page preset or custom page dimensions."""

    preset = str(raw.get("pagePreset", "a4_landscape"))
    if preset != "custom":
        return PAGE_PRESETS.get(preset, PAGE_PRESETS["a4_landscape"])
    width = clamp_int(raw.get("pageWidth"), 3508, 320, 6000)
    height = clamp_int(raw.get("pageHeight"), 2480, 320, 6000)
    return width, height


def normalize_custom_grid(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the editable custom grid payload from the browser."""

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
    """Normalize browser/API parameters into the internal rendering schema."""

    page_width, page_height = page_size_from_params(raw)
    layout_mode = str(raw.get("layoutMode", "grid"))
    if layout_mode not in VALID_LAYOUT_MODES:
        layout_mode = "grid"

    gradient_mode = str(raw.get("gradientMode", "accent"))
    if gradient_mode not in VALID_GRADIENT_MODES:
        gradient_mode = "accent"

    analysis_depth = str(raw.get("analysisDepth", "balanced")).lower()
    if analysis_depth not in VALID_ANALYSIS_DEPTHS:
        analysis_depth = "balanced"

    optimizer_mode = str(raw.get("bentoOptimizerMode", "balanced")).lower()
    if optimizer_mode not in VALID_BENTO_OPTIMIZER_MODES:
        optimizer_mode = "balanced"

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
        "bento_optimizer": parse_bool(raw.get("bentoOptimizer"), parse_bool(raw.get("autoImagesPerPage"), False)),
        "bento_optimizer_mode": optimizer_mode,
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
        "analysis_depth": analysis_depth,
    }
