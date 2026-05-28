#!/usr/bin/env python3
"""Scrape the local color-name database used by palette analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = BASE_DIR / "database" / "color_names.json"
NAME_THAT_COLOR_URL = "https://chir.ag/projects/ntc/ntc.js"
NAME_THAT_COLOR_PAGE = "https://chir.ag/projects/name-that-color/"


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "SP-Moodboard-ColorNames/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def rgb_from_hex(hex_value: str) -> list[int]:
    value = hex_value.strip().lstrip("#")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def scrape_name_that_color() -> list[dict[str, Any]]:
    source = fetch_text(NAME_THAT_COLOR_URL)
    colors: list[dict[str, Any]] = []
    for match in re.finditer(r'\["([0-9A-Fa-f]{6})",\s*"([^"]+)"\]', source):
        hex_value = "#" + match.group(1).upper()
        colors.append(
            {
                "name": match.group(2).strip(),
                "hex": hex_value,
                "rgb": rgb_from_hex(hex_value),
                "source": "Name That Color / Chirag Mehta",
            }
        )
    if not colors:
        raise RuntimeError("No color names were found in ntc.js")
    return colors


def dedupe_colors(colors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for color in colors:
        key = (str(color.get("name", "")).casefold(), str(color.get("hex", "")).upper())
        if key in seen:
            continue
        deduped.append(color)
        seen.add(key)
    return sorted(deduped, key=lambda item: (str(item["hex"]).upper(), str(item["name"]).casefold()))


def build_payload(colors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "color-names-v1",
        "generated": dt.date.today().isoformat(),
        "description": (
            "Local color-name database used for nearest-color palette naming. "
            "Pantone names are intentionally not redistributed."
        ),
        "sources": [
            {
                "name": "Name That Color",
                "url": NAME_THAT_COLOR_PAGE,
                "data_url": NAME_THAT_COLOR_URL,
                "license": "Creative Commons Attribution 2.5",
                "notes": (
                    "The upstream project states the color names were found via Wikipedia, "
                    "Crayola, and color-name dictionaries such as Resene."
                ),
            }
        ],
        "colors": colors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape local color-name data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    colors = dedupe_colors(scrape_name_that_color())
    payload = build_payload(colors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(colors)} color names to {args.output}")


if __name__ == "__main__":
    main()
