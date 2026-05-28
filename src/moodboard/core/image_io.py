"""Low-level image constants and helpers.

Keep this module free of ML imports. It should stay safe to import from CLI
tools, dataset adapters and tests.
"""

from __future__ import annotations

import colorsys
import io
import math
import sys

from PIL import Image, ImageOps, ImageStat

from .schemas import ImageInfo, UploadedImage


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow fallback
    RESAMPLE = Image.LANCZOS


def classify_orientation_from_ratio(ratio: float, threshold: float) -> str:
    if ratio > threshold:
        return "landscape"
    if ratio < 1.0 / threshold:
        return "portrait"
    return "square"


def classify_orientation(width: int, height: int, threshold: float) -> str:
    if height <= 0:
        return "square"
    return classify_orientation_from_ratio(width / height, threshold)


def open_asset_rgb(asset: UploadedImage) -> Image.Image:
    """Open an uploaded image as RGB after applying EXIF orientation."""

    with Image.open(io.BytesIO(asset.data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def analyze_images(assets: list[UploadedImage], orientation_threshold: float) -> list[ImageInfo]:
    """Extract lightweight image metadata used by analysis and Bento layout."""

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
    """Sort analyzed images for color-gradient and hero-based page ordering."""

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
