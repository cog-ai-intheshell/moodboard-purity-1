"""Bento composition optimizer.

The first production version is deterministic on purpose: it evaluates many
candidate page densities, image orders and Bento geometries, then picks the
highest-scoring plan. The scoring function is kept explicit so a trained Bento
ranker can later replace or reweight it without changing the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Any

from src.moodboard.bento.layout import BENTO_GRIDS, GRID_SLOT_COUNT, build_auto_norm_rects, crop_grid_for_images
from src.moodboard.core.image_io import classify_orientation_from_ratio, sort_infos
from src.moodboard.core.schemas import ImageInfo


@dataclass(frozen=True)
class BentoPageSpec:
    infos: list[ImageInfo]
    rects: list[tuple[float, float, float, float]]
    layout_name: str
    score: float
    details: dict[str, float]


@dataclass(frozen=True)
class BentoPlan:
    pages: list[BentoPageSpec]
    images_per_page: int
    score: float
    mode: str
    rationale: dict[str, float]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return low
    return max(low, min(high, value))


def circular_distance(left: float, right: float) -> float:
    diff = abs((left % 1.0) - (right % 1.0))
    return min(diff, 1.0 - diff) * 2.0


def visual_distance(left: ImageInfo, right: ImageInfo) -> float:
    left_h, left_s, left_v = left.hsv
    right_h, right_s, right_v = right.hsv
    hue = circular_distance(left.accent_h or left_h, right.accent_h or right_h)
    saturation = abs(left_s - right_s)
    value = abs(left_v - right_v)
    brightness = abs(left.brightness - right.brightness)
    contrast = abs(left.contrast - right.contrast) * 0.5
    orientation = 0.0 if left.orientation == right.orientation else 0.35
    return clamp(hue * 0.35 + saturation * 0.16 + value * 0.18 + brightness * 0.16 + contrast * 0.10 + orientation * 0.05)


def visual_bucket(info: ImageInfo) -> tuple[int, int, int]:
    hue = int(((info.accent_h or info.hsv[0]) % 1.0) * 8)
    value = 0 if info.brightness < 0.35 else (2 if info.brightness > 0.72 else 1)
    saturation = 0 if info.hsv[1] < 0.20 else (2 if info.hsv[1] > 0.58 else 1)
    return hue, value, saturation


def image_ratio(info: ImageInfo) -> float:
    return max(0.05, info.width / max(1.0, float(info.height)))


def rect_ratio(rect: tuple[float, float, float, float], page_size: tuple[int, int]) -> float:
    page_width, page_height = page_size
    width = max(0.001, (rect[2] - rect[0]) * page_width)
    height = max(0.001, (rect[3] - rect[1]) * page_height)
    return width / height


def rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def orientation_for_rect(rect: tuple[float, float, float, float], params: dict[str, Any]) -> str:
    return classify_orientation_from_ratio(rect_ratio(rect, params["page_size"]), float(params.get("orientation_threshold", 1.2)))


def crop_fit_score(info: ImageInfo, rect: tuple[float, float, float, float], params: dict[str, Any]) -> float:
    ratio_delta = abs(math.log(image_ratio(info) / rect_ratio(rect, params["page_size"])))
    score = math.exp(-ratio_delta * 0.92)
    if info.orientation == orientation_for_rect(rect, params):
        score = score * 0.82 + 0.18
    elif info.orientation == "square" or orientation_for_rect(rect, params) == "square":
        score *= 0.86
    else:
        score *= 0.58
    if params.get("fit_mode") == "contain":
        score = score * 0.70 + 0.30
    return clamp(score)


def simple_grid_rects(count: int) -> list[tuple[float, float, float, float]]:
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    rects = []
    for idx in range(count):
        row = idx // cols
        col = idx % cols
        rects.append((col / cols, row / rows, (col + 1) / cols, (row + 1) / rows))
    return rects


def normalized(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-9:
        return [0.65 for _value in values]
    return [(value - lo) / (hi - lo) for value in values]


def assign_infos_to_rects(
    batch: list[ImageInfo],
    rects: list[tuple[float, float, float, float]],
    params: dict[str, Any],
) -> list[ImageInfo]:
    """Assign images to slots by balancing hero strength and crop fit."""

    remaining = list(batch)
    slots = list(enumerate(rects))
    slot_areas = [rect_area(rect) for rect in rects]
    hero_slots = sorted(slots, key=lambda item: rect_area(item[1]), reverse=True)
    hero_values = normalized([info.hero_score for info in batch])
    hero_by_name = {info.asset.filename: hero for info, hero in zip(batch, hero_values)}
    assigned: list[ImageInfo | None] = [None] * len(rects)

    for slot_idx, rect in hero_slots:
        if not remaining:
            break
        area_weight = rect_area(rect) / max(slot_areas or [1.0])
        best_idx = 0
        best_score = -1.0
        for idx, info in enumerate(remaining):
            fit = crop_fit_score(info, rect, params)
            hero = hero_by_name.get(info.asset.filename, 0.5)
            score = fit * 0.64 + hero * (0.24 + area_weight * 0.20)
            if info.orientation == orientation_for_rect(rect, params):
                score += 0.08
            if score > best_score:
                best_idx = idx
                best_score = score
        assigned[slot_idx] = remaining.pop(best_idx)

    fallback = batch[0]
    return [info or fallback for info in assigned]


def grid_candidates(slot_count: int, params: dict[str, Any], custom_grid: dict[str, Any] | None) -> list[dict[str, Any]]:
    layout_mode = str(params.get("layout_mode", "grid"))
    candidates: list[dict[str, Any]] = []

    if layout_mode == "custom" and custom_grid is not None:
        grid = crop_grid_for_images(custom_grid, min(slot_count, len(custom_grid["rects"])))
        rects = grid["rects"]
        if rects:
            candidates.append({"name": "custom optimized", "rects": rects})
        return candidates

    if layout_mode == "simple":
        return [{"name": "simple optimized", "rects": simple_grid_rects(slot_count)}]

    if layout_mode in {"grid", "random"}:
        for grid in BENTO_GRIDS:
            cropped = crop_grid_for_images(grid, min(slot_count, GRID_SLOT_COUNT))
            candidates.append({"name": cropped["name"], "rects": cropped["rects"]})
        return candidates

    if layout_mode == "auto":
        return [{"name": "organic optimized", "rects": build_auto_norm_rects(slot_count)}]

    for grid in BENTO_GRIDS:
        cropped = crop_grid_for_images(grid, min(slot_count, GRID_SLOT_COUNT))
        candidates.append({"name": cropped["name"], "rects": cropped["rects"]})
    candidates.append({"name": "organic optimized", "rects": build_auto_norm_rects(slot_count)})
    return candidates


def candidate_densities(total: int, params: dict[str, Any], custom_grid: dict[str, Any] | None) -> list[int]:
    if total <= 0:
        return []
    layout_mode = str(params.get("layout_mode", "grid"))
    mode = str(params.get("bento_optimizer_mode", "balanced"))
    manual = int(params.get("images_per_page", 12))
    if layout_mode == "custom" and custom_grid is not None:
        upper = min(total, max(1, len(custom_grid["rects"])))
    elif layout_mode in {"grid", "random"}:
        upper = min(total, GRID_SLOT_COUNT)
    elif layout_mode == "simple":
        upper = min(total, 30)
    else:
        upper = min(total, 20)

    if upper <= 3:
        return list(range(1, upper + 1))

    if mode == "editorial":
        base = [4, 5, 6, 7, 8, 9, 10]
    elif mode == "dense":
        base = [8, 10, 12, 14, 16, 18, 20]
    elif mode == "clustered":
        base = [5, 6, 8, 9, 10, 12, 14]
    else:
        base = [6, 8, 9, 10, 12, 14, 16]

    around = [manual - 2, manual - 1, manual, manual + 1, manual + 2]
    values = sorted({value for value in base + around + [upper] if 1 <= value <= upper})
    return values or [upper]


def ordering_variants(infos: list[ImageInfo], params: dict[str, Any]) -> list[tuple[str, list[ImageInfo]]]:
    variants: list[tuple[str, list[ImageInfo]]] = [
        ("input", list(infos)),
        ("hero", sorted(infos, key=lambda info: -info.hero_score)),
        ("clustered-color", sorted(infos, key=lambda info: (visual_bucket(info), info.accent_h, -info.hero_score))),
        ("dark-light", sorted(infos, key=lambda info: (info.brightness, info.accent_h))),
        ("contrast", sorted(infos, key=lambda info: (-info.contrast, info.brightness, info.accent_h))),
    ]
    if params.get("use_color_gradient"):
        variants.append((f"gradient-{params.get('gradient_mode', 'accent')}", sort_infos(infos, str(params.get("gradient_mode", "accent")))))

    mode = str(params.get("bento_optimizer_mode", "balanced"))
    if mode == "clustered":
        variants.insert(0, ("cluster-priority", sorted(infos, key=lambda info: (visual_bucket(info), -info.hero_score))))
    if mode == "editorial":
        variants.insert(0, ("editorial-hero", sorted(infos, key=lambda info: (-info.hero_score, info.accent_h))))
    if mode == "dense":
        variants.insert(0, ("dense-flow", sorted(infos, key=lambda info: (info.orientation, info.accent_h, info.brightness))))

    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[str, list[ImageInfo]]] = []
    for name, ordered in variants:
        signature = tuple(info.asset.filename for info in ordered)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append((name, ordered))
    return unique


def adjacency_pairs(rects: list[tuple[float, float, float, float]]) -> list[tuple[int, int, float]]:
    centers = [((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0) for rect in rects]
    pairs: list[tuple[int, int, float]] = []
    for idx, left in enumerate(centers):
        distances = []
        for other_idx, right in enumerate(centers):
            if idx == other_idx:
                continue
            distance = math.hypot(left[0] - right[0], left[1] - right[1])
            distances.append((distance, other_idx))
        for distance, other_idx in sorted(distances)[:3]:
            if idx < other_idx:
                pairs.append((idx, other_idx, distance))
    return pairs


def target_density_score(per_page: int, mode: str) -> float:
    targets = {
        "editorial": (6.0, 2.4),
        "balanced": (10.0, 3.2),
        "dense": (16.0, 4.0),
        "clustered": (9.0, 3.0),
    }
    target, spread = targets.get(mode, targets["balanced"])
    return math.exp(-((per_page - target) ** 2) / max(1.0, 2.0 * spread * spread))


def hierarchy_score(rects: list[tuple[float, float, float, float]], mode: str) -> float:
    areas = sorted([rect_area(rect) for rect in rects], reverse=True)
    if not areas:
        return 0.0
    if len(areas) == 1:
        return 0.85
    median = sorted(areas)[len(areas) // 2]
    ratio = areas[0] / max(0.001, median)
    ideal = {"editorial": 2.8, "balanced": 2.1, "dense": 1.45, "clustered": 1.85}.get(mode, 2.1)
    return math.exp(-abs(math.log(ratio / ideal)) * 0.8)


def score_page(
    infos: list[ImageInfo],
    rects: list[tuple[float, float, float, float]],
    params: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    if not infos or not rects:
        return 0.0, {}

    mode = str(params.get("bento_optimizer_mode", "balanced"))
    pairs = adjacency_pairs(rects)
    crop_scores = [crop_fit_score(info, rect, params) for info, rect in zip(infos, rects)]
    crop = mean(crop_scores) if crop_scores else 0.0

    areas = [rect_area(rect) for rect in rects]
    hero_values = normalized([info.hero_score for info in infos])
    ideal = sum(area * hero for area, hero in zip(sorted(areas, reverse=True), sorted(hero_values, reverse=True)))
    actual = sum(area * hero for area, hero in zip(areas, hero_values))
    hero = clamp(actual / max(ideal, 1e-9))

    if pairs:
        distances = [visual_distance(infos[left], infos[right]) for left, right, _distance in pairs]
        average_distance = mean(distances)
        target_distance = {"editorial": 0.46, "balanced": 0.34, "dense": 0.26, "clustered": 0.18}.get(mode, 0.34)
        flow = math.exp(-abs(average_distance - target_distance) * 2.2)
        local_coherence = 1.0 - clamp(average_distance)
    else:
        flow = 0.8
        local_coherence = 0.8

    brightness_values = [info.brightness for info in infos]
    saturation_values = [info.hsv[1] for info in infos]
    contrast_values = [info.contrast for info in infos]
    brightness_balance = math.exp(-abs(mean(brightness_values) - 0.54) * 1.8)
    brightness_variety = clamp((pstdev(brightness_values) if len(brightness_values) > 1 else 0.16) / 0.26)
    saturation_balance = math.exp(-abs(mean(saturation_values) - 0.38) * 1.2)
    contrast_presence = clamp(mean(contrast_values) / 0.72)
    visual_balance = clamp(brightness_balance * 0.38 + brightness_variety * 0.24 + saturation_balance * 0.20 + contrast_presence * 0.18)

    cluster = local_coherence if mode == "clustered" else clamp(local_coherence * 0.68 + flow * 0.32)
    hierarchy = hierarchy_score(rects, mode)
    density = target_density_score(len(infos), mode)

    score = (
        crop * 0.26
        + hero * 0.17
        + flow * 0.15
        + visual_balance * 0.15
        + cluster * 0.12
        + hierarchy * 0.09
        + density * 0.06
    )
    return clamp(score), {
        "crop": crop,
        "hero": hero,
        "flow": flow,
        "balance": visual_balance,
        "cluster": cluster,
        "hierarchy": hierarchy,
        "density": density,
    }


def score_plan(pages: list[BentoPageSpec], mode: str, total: int) -> tuple[float, dict[str, float]]:
    if not pages:
        return 0.0, {}
    average = mean(page.score for page in pages)
    last_page_fill = len(pages[-1].infos) / max(1, pages[0].infos and len(pages[0].infos) or 1)
    single_or_full_last = 1.0 if len(pages) == 1 or last_page_fill >= 0.45 else 0.72
    page_penalty = 1.0 - min(0.10, max(0, len(pages) - math.ceil(total / max(1, len(pages[0].infos)))) * 0.018)
    score = clamp(average * 0.88 + single_or_full_last * 0.08 + page_penalty * 0.04)

    keys = sorted({key for page in pages for key in page_score_details(page).keys()})
    details = {key: mean(page_score_details(page).get(key, 0.0) for page in pages) for key in keys}
    details["plan"] = score
    details["pageFill"] = single_or_full_last
    details["pageCount"] = float(len(pages))
    details["mode"] = 0.0 if mode == "balanced" else (1.0 if mode == "dense" else 2.0 if mode == "editorial" else 3.0)
    return score, details


def page_score_details(page: BentoPageSpec) -> dict[str, float]:
    return page.details


def optimize_bento_plan(
    infos: list[ImageInfo],
    params: dict[str, Any],
    custom_grid: dict[str, Any] | None = None,
) -> BentoPlan:
    """Return the highest-scoring Bento plan for the current image set."""

    if not infos:
        return BentoPlan([], 0, 0.0, str(params.get("bento_optimizer_mode", "balanced")), {})

    mode = str(params.get("bento_optimizer_mode", "balanced"))
    best_plan: BentoPlan | None = None

    for per_page in candidate_densities(len(infos), params, custom_grid):
        for order_name, ordered_infos in ordering_variants(infos, params):
            plan_pages: list[BentoPageSpec] = []
            valid = True
            for start in range(0, len(ordered_infos), per_page):
                batch = ordered_infos[start : start + per_page]
                if not batch:
                    continue
                best_page: BentoPageSpec | None = None
                for grid in grid_candidates(len(batch), params, custom_grid):
                    rects = list(grid["rects"])[: len(batch)]
                    if len(rects) != len(batch):
                        continue
                    assigned = assign_infos_to_rects(batch, rects, params)
                    page_score, details = score_page(assigned, rects, params)
                    details["order"] = float(abs(hash(order_name)) % 1000) / 1000.0
                    page = BentoPageSpec(assigned, rects, str(grid["name"]), page_score, details)
                    if best_page is None or page.score > best_page.score:
                        best_page = page
                if best_page is None:
                    valid = False
                    break
                plan_pages.append(best_page)
            if not valid or not plan_pages:
                continue
            plan_score, rationale = score_plan(plan_pages, mode, len(infos))
            candidate = BentoPlan(plan_pages, per_page, plan_score, mode, rationale)
            if best_plan is None or candidate.score > best_plan.score:
                best_plan = candidate

    if best_plan is not None:
        return best_plan

    rects = build_auto_norm_rects(min(len(infos), max(1, int(params.get("images_per_page", 12)))))
    assigned = assign_infos_to_rects(infos[: len(rects)], rects, params)
    page_score, details = score_page(assigned, rects, params)
    page = BentoPageSpec(assigned, rects, "organic fallback", page_score, details)
    return BentoPlan([page], len(assigned), page_score, mode, details)
