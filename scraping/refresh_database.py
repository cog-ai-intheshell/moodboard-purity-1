#!/usr/bin/env python3
"""Refresh local metadata databases used by the moodboard pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scraping.aesthetic_sources import scrape_aesthetic_source_items
from scraping.scrape_color_names import build_payload, dedupe_colors, scrape_name_that_color

DATABASE_DIR = BASE_DIR / "database"
AESTHETICS_CACHE_PATH = DATABASE_DIR / "aesthetics_cache.json"
COLOR_NAMES_PATH = DATABASE_DIR / "color_names.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_colors(output: Path = COLOR_NAMES_PATH) -> dict[str, Any]:
    colors = dedupe_colors(scrape_name_that_color())
    payload = build_payload(colors)
    write_json(output, payload)
    return {"kind": "colors", "path": str(output.relative_to(BASE_DIR)), "count": len(colors)}


def refresh_aesthetics(
    output: Path = AESTHETICS_CACHE_PATH,
    *,
    limit: int = 500,
    source: str = "all",
    allow_empty: bool = False,
) -> dict[str, Any]:
    payload = scrape_aesthetic_source_items(limit, source=source)
    aesthetics = payload.get("aesthetics") if isinstance(payload, dict) else []
    total = len(aesthetics) if isinstance(aesthetics, list) else 0
    if total == 0 and not allow_empty:
        raise RuntimeError("Aesthetic scrape returned no items; use --allow-empty to overwrite anyway.")
    cache = {
        **payload,
        "created": dt.datetime.now(dt.timezone.utc).timestamp(),
        "generated": dt.date.today().isoformat(),
        "total_raw": total,
    }
    write_json(output, cache)
    return {
        "kind": "aesthetics",
        "path": str(output.relative_to(BASE_DIR)),
        "count": total,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh local moodboard databases.")
    parser.add_argument("--colors", action="store_true", help="Refresh database/color_names.json.")
    parser.add_argument("--aesthetics", action="store_true", help="Refresh database/aesthetics_cache.json.")
    parser.add_argument("--source", default="all", choices=("all", "wiki", "cari"), help="Aesthetic source to scrape.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum aesthetics to request per source.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow writing an empty aesthetics cache.")
    args = parser.parse_args()

    refresh_all = not args.colors and not args.aesthetics
    results: list[dict[str, Any]] = []
    if refresh_all or args.colors:
        results.append(refresh_colors())
    if refresh_all or args.aesthetics:
        results.append(refresh_aesthetics(limit=args.limit, source=args.source, allow_empty=args.allow_empty))
    for result in results:
        print(f"Refreshed {result['kind']}: {result['count']} records -> {result['path']}")


if __name__ == "__main__":
    main()
