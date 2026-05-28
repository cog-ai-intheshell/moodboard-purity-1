"""Bento grid layout selection and slot assignment.

This module owns the geometry of Bento pages: predefined grid matrices, custom
grid normalization, automatic recursive splits and the matching of images to
slots. The actual image rendering stays in `render.py`/the legacy facade while
the migration continues.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.moodboard.core.image_io import classify_orientation_from_ratio
from src.moodboard.core.schemas import ImageInfo


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
    """Parse and clamp integer layout inputs coming from the browser."""

    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def build_bento_grids() -> tuple[list[dict[str, Any]], int]:
    """Compile symbolic grid matrices into normalized rectangles."""

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
    """Recursively split a normalized rectangle into `count` balanced cells."""

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
    """Build an auto-layout grid for arbitrary image counts."""

    rects = split_rect_equally((0.0, 0.0, 1.0, 1.0), count)
    return sorted(rects, key=lambda rc: ((rc[1] + rc[3]) / 2.0, (rc[0] + rc[2]) / 2.0))


def auto_images_per_page(total_images: int) -> int:
    """Pick a reasonable page density when the UI is in automatic mode."""

    if total_images <= 0:
        return 0
    guess = int(math.sqrt(total_images) * 3)
    return max(6, min(20, guess))


def choose_best_grid_for_batch(batch_infos: list[ImageInfo]) -> dict[str, Any]:
    """Choose the predefined grid whose slot orientations best match a batch."""

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
    """Crop a larger grid down to the number of images on the current page."""

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
    """Convert the editable 6x4 UI grid into normalized Bento rectangles."""

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
    """Place high-scoring images into hero slots, then fill remaining slots."""

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
