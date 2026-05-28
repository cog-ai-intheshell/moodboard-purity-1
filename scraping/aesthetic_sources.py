#!/usr/bin/env python3
"""Scrape aesthetic metadata from Aesthetics Wiki and CARI."""

from __future__ import annotations

import html as html_lib
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


CARI_INDEX_URL = "https://cari.institute/aesthetics"
CARI_SITEMAP_URL = "https://cari.institute/sitemap.xml"
AESTHETICS_WIKI_API_URL = "https://aesthetics.fandom.com/api.php"
MAX_AESTHETIC_SCRAPE_ITEMS = 5000


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def parse_aesthetic_limit(value: Any, default: int = 500) -> int:
    text = str(value or "").strip().lower()
    if text in {"all", "full", "max", "*", "0"}:
        return MAX_AESTHETIC_SCRAPE_ITEMS
    return clamp_int(value, default, 20, MAX_AESTHETIC_SCRAPE_ITEMS)


def wiki_api_get(params: dict[str, Any], timeout: int = 18) -> dict[str, Any]:
    query = dict(params)
    query.setdefault("format", "json")
    query.setdefault("formatversion", "2")
    url = f"{AESTHETICS_WIKI_API_URL}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "MoodboardCognitiveMVP/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, timeout: int = 24) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MoodboardCognitiveMVP/1.0 (+metadata-only aesthetic research)",
            "Accept": "text/html,application/xml,text/xml,application/json;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return data.decode(charset, errors="ignore")


def html_to_plain_text(markup: str) -> str:
    text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", markup, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|h[1-6]|li|div|section|article|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def slug_to_title(slug: str) -> str:
    words = re.sub(r"[-_]+", " ", urllib.parse.unquote(slug)).strip()
    small_words = {"a", "an", "and", "as", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}
    parts = []
    for index, word in enumerate(words.split()):
        lower = word.lower()
        if index > 0 and lower in small_words:
            parts.append(lower)
        else:
            parts.append(word[:1].upper() + word[1:])
    return " ".join(parts)


def title_to_wiki_url(title: str) -> str:
    return f"https://aesthetics.fandom.com/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def infer_aesthetic_metadata(title: str, description: str) -> dict[str, list[str]]:
    text = f"{title} {description}".lower()
    vocab = {
        "colors": {
            "black", "white", "ivory", "cream", "red", "orange", "gold", "yellow", "green", "sage",
            "emerald", "cyan", "blue", "sky", "purple", "violet", "lavender", "pink", "brown", "gray",
            "silver", "pastel", "neon", "burgundy",
        },
        "emotions": {
            "melancholy", "spirituality", "violence", "nostalgia", "serenity", "majesty", "strangeness",
            "mysticism", "chaos", "innocence", "liminality", "uncanny", "calm", "dream", "dread",
        },
        "symbols": {
            "eye", "eyes", "star", "stars", "halo", "spiral", "horse", "dragon", "lion", "light",
            "cosmos", "architecture", "weapon", "weapons", "church", "cross", "moon", "flower",
            "flowers", "forest", "water", "city", "screen", "book", "books", "angel", "wings",
        },
        "styles": {
            "watercolor", "pencil", "painting", "manga", "comics", "digital", "engraving", "vector",
            "pixel", "photography", "gothic", "surreal", "liminal", "glossy", "ornamental", "organic",
            "minimal", "retro", "fantasy",
        },
    }
    return {
        key: sorted({term for term in terms if re.search(rf"\b{re.escape(term)}\b", text)})[:10]
        for key, terms in vocab.items()
    }


def wiki_plain_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"&nbsp;|&amp;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wiki_field(content: str, field: str) -> str:
    pattern = r"\|" + re.escape(field) + r"\s*=\s*(.*?)(?=\n\||\n\}\}|\Z)"
    match = re.search(pattern, content, flags=re.S | re.I)
    return wiki_plain_text(match.group(1)) if match else ""


def wiki_field_raw(content: str, field: str) -> str:
    pattern = r"\|" + re.escape(field) + r"\s*=\s*(.*?)(?=\n\||\n\}\}|\Z)"
    match = re.search(pattern, content, flags=re.S | re.I)
    return match.group(1).strip() if match else ""


def wiki_summary(content: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", " ", content)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("|", "!", "=", "[[Category:", "[[File:", "__")):
            continue
        plain = wiki_plain_text(line)
        if plain and not plain.startswith(("Category:", "File:")):
            lines.append(plain)
        if len(" ".join(lines)) > 900:
            break
    return " ".join(lines)[:1400].strip()


def wiki_related(content: str) -> list[str]:
    related = wiki_field_raw(content, "related_aesthetics")
    names = set()
    for target, label in re.findall(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]", related):
        names.add(wiki_plain_text(label or target))
    return sorted(name for name in names if name)[:14]


def should_skip_wiki_title(title: str) -> bool:
    lower = title.lower().strip()
    if not lower or ":" in lower:
        return True
    skip_prefixes = (
        "aesthetics wiki",
        "list of",
        "help:",
        "rules",
        "wiki ",
        "category:",
        "template:",
        "file:",
        "user:",
    )
    skip_exact = {
        "home",
        "main page",
        "about",
        "faq",
        "frequently asked questions",
        "wanted pages",
        "deleted pages",
        "recent changes",
        "random page",
    }
    return lower in skip_exact or lower.startswith(skip_prefixes)


def wiki_all_titles(limit: int) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    query: dict[str, Any] = {
        "action": "query",
        "list": "allpages",
        "apnamespace": "0",
        "aplimit": "max",
    }
    while len(titles) < limit:
        payload = wiki_api_get(query)
        for item in payload.get("query", {}).get("allpages", []) or []:
            title = str(item.get("title", "")).strip()
            if title in seen or should_skip_wiki_title(title):
                continue
            seen.add(title)
            titles.append(title)
            if len(titles) >= limit:
                break
        continuation = payload.get("continue", {})
        if not continuation or len(titles) >= limit:
            break
        query.update(continuation)
    return titles


def wiki_list_titles(limit: int) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    query: dict[str, Any] = {
        "action": "query",
        "prop": "links",
        "titles": "List of Aesthetics",
        "pllimit": "max",
    }
    while len(titles) < limit:
        payload = wiki_api_get(query)
        for page in payload.get("query", {}).get("pages", []) or []:
            for link in page.get("links", []) or []:
                title = str(link.get("title", "")).strip()
                if title in seen or should_skip_wiki_title(title):
                    continue
                seen.add(title)
                titles.append(title)
                if len(titles) >= limit:
                    break
            if len(titles) >= limit:
                break
        continuation = payload.get("continue", {})
        if not continuation or len(titles) >= limit:
            break
        query.update(continuation)
    return titles


def wiki_pages_from_titles(titles: list[str]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for start in range(0, len(titles), 35):
        batch = titles[start : start + 35]
        payload = wiki_api_get(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "redirects": "1",
                "titles": "|".join(batch),
            },
            timeout=28,
        )
        pages.extend(payload.get("query", {}).get("pages", []) or [])
    return pages


def wiki_page_to_aesthetic(page: dict[str, Any]) -> dict[str, Any] | None:
    title = str(page.get("title", "")).strip()
    if not title or should_skip_wiki_title(title):
        return None
    revisions = page.get("revisions", []) or []
    content = ""
    if revisions:
        content = str(revisions[0].get("slots", {}).get("main", {}).get("content", ""))
    extract = str(page.get("extract", "") or wiki_summary(content)).strip()
    motifs = wiki_field(content, "key_motifs")
    values = wiki_field(content, "key_values")
    color_field = wiki_field(content, "key_colours") or wiki_field(content, "key_colors")
    media = wiki_field(content, "related_media")
    metadata = infer_aesthetic_metadata(title, " ".join([extract, motifs, values, color_field, media]))
    if color_field:
        metadata["colors"] = sorted(set(metadata["colors"] + re.findall(r"[A-Za-z][A-Za-z -]{2,}", color_field.lower())))[:14]
    return {
        "name": title,
        "description": extract[:1600],
        "colors": metadata["colors"],
        "emotions": metadata["emotions"],
        "symbols": metadata["symbols"],
        "styles": metadata["styles"],
        "tags": sorted(set(metadata["colors"] + metadata["emotions"] + metadata["symbols"] + metadata["styles"]))[:24],
        "related": wiki_related(content),
        "motifs": motifs[:720],
        "values": values[:720],
        "era": wiki_field(content, "decade_of_origin")[:180],
        "media": media[:360],
        "source": "Aesthetics Wiki",
        "source_url": title_to_wiki_url(title),
    }


def scrape_aesthetics_wiki_items(limit: int = 500) -> list[dict[str, Any]]:
    max_items = parse_aesthetic_limit(limit, 500)
    list_titles = wiki_list_titles(max_items)
    all_titles = wiki_all_titles(max_items)
    titles: list[str] = []
    seen: set[str] = set()
    for title in [*list_titles, *all_titles]:
        if title not in seen:
            seen.add(title)
            titles.append(title)
        if len(titles) >= max_items:
            break
    aesthetics = []
    for page in wiki_pages_from_titles(titles):
        item = wiki_page_to_aesthetic(page)
        if item:
            aesthetics.append(item)
    return aesthetics


def discover_cari_aesthetic_urls(limit: int) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def add_url(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc and parsed.netloc != "cari.institute":
            return
        path = parsed.path.rstrip("/")
        if not path.startswith("/aesthetics/"):
            return
        slug = path.rsplit("/", 1)[-1]
        if not slug or slug in {"aesthetics", "categories"}:
            return
        full_url = f"https://cari.institute{path}"
        if full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)

    try:
        sitemap = fetch_text(CARI_SITEMAP_URL)
        root = ET.fromstring(sitemap)
        for loc in root.findall(".//{*}loc"):
            if loc.text:
                add_url(loc.text.strip())
                if len(urls) >= limit:
                    return urls
    except Exception as exc:
        print(f"[WARN] CARI sitemap unavailable: {exc}", file=sys.stderr)

    try:
        index_html = fetch_text(CARI_INDEX_URL)
        for match in re.findall(r"href=[\"']([^\"']*/aesthetics/[^\"'#?]+)", index_html, flags=re.I):
            add_url(urllib.parse.urljoin(CARI_INDEX_URL, match))
            if len(urls) >= limit:
                break
    except Exception as exc:
        print(f"[WARN] CARI index discovery unavailable: {exc}", file=sys.stderr)
    return urls[:limit]


def cari_page_to_aesthetic(url: str, markup: str) -> dict[str, Any] | None:
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", markup, flags=re.S | re.I)
    title = html_to_plain_text(title_match.group(1)).strip() if title_match else ""
    if not title:
        slug = urllib.parse.urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
        title = slug_to_title(slug)
    if not title or title.lower() in {"aesthetics", "index of aesthetics"}:
        return None

    text = html_to_plain_text(markup)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start = 0
    for index, line in enumerate(lines):
        if line.lower() == title.lower():
            start = index + 1
            break
    description_lines = []
    for line in lines[start:]:
        lower = line.lower()
        if lower.startswith(("links", "timeline", "period of relevance", "sorted decade", "attribution", "gallery", "related aesthetics", "©")):
            break
        if lower in {"visit website", "show more", "menu", "close menu", "title", "description"}:
            continue
        if line in {"* * *", "---"} or line.startswith("Image:"):
            continue
        description_lines.append(line)
        if len(" ".join(description_lines)) > 1400:
            break
    description = " ".join(description_lines).strip()
    if not description:
        description = " ".join(lines[start : start + 4])[:1000].strip()

    period = ""
    sorted_decade = ""
    period_match = re.search(r"Period of relevance:\s*([^\n]+)", text, flags=re.I)
    decade_match = re.search(r"Sorted decade:\s*([^\n]+)", text, flags=re.I)
    if period_match:
        period = period_match.group(1).strip()[:180]
    if decade_match:
        sorted_decade = decade_match.group(1).strip()[:80]
    alternate_match = re.search(r"Alternate names:\s*([^\n]+)", text, flags=re.I)
    alternate_names = []
    if alternate_match:
        alternate_names = [value.strip() for value in re.split(r",|;", alternate_match.group(1)) if value.strip()][:12]

    related = []
    for href, label in re.findall(r"<a[^>]+href=[\"']([^\"']*/aesthetics/[^\"'#?]+)[\"'][^>]*>(.*?)</a>", markup, flags=re.S | re.I):
        parsed = urllib.parse.urlparse(urllib.parse.urljoin(url, href))
        related_name = html_to_plain_text(label).strip() or slug_to_title(parsed.path.rstrip("/").rsplit("/", 1)[-1])
        if related_name and related_name.lower() != title.lower() and related_name not in related:
            related.append(related_name)

    metadata = infer_aesthetic_metadata(title, " ".join([description, period, sorted_decade, " ".join(alternate_names)]))
    return {
        "name": title,
        "description": description[:1600],
        "colors": metadata["colors"],
        "emotions": metadata["emotions"],
        "symbols": metadata["symbols"],
        "styles": metadata["styles"],
        "tags": sorted(set(metadata["colors"] + metadata["emotions"] + metadata["symbols"] + metadata["styles"] + alternate_names))[:24],
        "related": related[:18],
        "motifs": "",
        "values": "",
        "era": period or sorted_decade,
        "media": "",
        "alternate_names": alternate_names,
        "source": "CARI Institute",
        "source_url": url,
        "asset_policy": "metadata-only; no CARI image or asset redistribution",
    }


def scrape_cari_aesthetic_items(limit: int = 500) -> list[dict[str, Any]]:
    max_items = parse_aesthetic_limit(limit, 500)
    items: list[dict[str, Any]] = []
    for url in discover_cari_aesthetic_urls(max_items):
        try:
            item = cari_page_to_aesthetic(url, fetch_text(url))
        except Exception as exc:
            print(f"[WARN] Cannot scrape CARI page {url}: {exc}", file=sys.stderr)
            continue
        if item:
            items.append(item)
    return items


def scrape_aesthetic_source_items(limit: int = 500, source: str = "all") -> dict[str, Any]:
    selected = str(source or "all").lower()
    max_items = parse_aesthetic_limit(limit, 500)
    aesthetics: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    if selected in {"all", "wiki", "aesthetics-wiki", "aesthetics_wiki"}:
        try:
            wiki_items = scrape_aesthetics_wiki_items(max_items)
            aesthetics.extend(wiki_items)
            counts["aesthetics_wiki"] = len(wiki_items)
        except Exception as exc:
            counts["aesthetics_wiki"] = 0
            print(f"[WARN] Aesthetics Wiki scrape failed: {exc}", file=sys.stderr)

    if selected in {"all", "cari", "cari-institute", "cari_institute"}:
        try:
            cari_items = scrape_cari_aesthetic_items(max_items)
            aesthetics.extend(cari_items)
            counts["cari_institute"] = len(cari_items)
        except Exception as exc:
            counts["cari_institute"] = 0
            print(f"[WARN] CARI scrape failed: {exc}", file=sys.stderr)

    return {
        "source": "Aesthetics Wiki MediaWiki API + CARI Institute metadata pages" if selected == "all" else selected,
        "source_urls": [AESTHETICS_WIKI_API_URL, CARI_INDEX_URL],
        "counts": counts,
        "policy": {
            "cari": "metadata-only; no CARI image or asset redistribution in MVP",
            "aesthetics_wiki": "text metadata for aesthetic retrieval; no image redistribution",
        },
        "aesthetics": aesthetics,
    }


def scrape_aesthetics_wiki(limit: int = 500) -> dict[str, Any]:
    return scrape_aesthetic_source_items(limit, source="wiki")
