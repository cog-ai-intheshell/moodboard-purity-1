#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Normalize local/filtered dataset samples into the world-model JSONL schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.moodboard.ai.models.colors.palette_extractor import image_palette
from src.moodboard.ai.variables.labels import clean_concept_label
from src.moodboard.ai.orchestrator import analyze
from src.moodboard.core.image_io import VALID_EXTENSIONS, classify_orientation
from src.moodboard.core.paths import BASE_DIR, WORLD_SAMPLES_PATH
from src.moodboard.core.schemas import UploadedImage, VALID_ANALYSIS_DEPTHS, normalize_params


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def image_metrics(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    width, height = rgb.size
    orientation = classify_orientation(width, height, 1.2)
    gray = rgb.convert("L")
    gray_stat = ImageStat.Stat(gray)
    gray.close()
    return {
        "width": width,
        "height": height,
        "orientation": orientation,
        "brightness": round(gray_stat.mean[0] / 255.0, 4),
        "mean_rgb": [round(value, 2) for value in stat.mean],
    }


def unique_terms(values: list[Any], limit: int = 24) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = clean_concept_label(str(value), max_words=5)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= limit:
            break
    return out


def semantic_enrichment_for_paths(
    paths: list[Path],
    analysis_depth: str,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    if not paths:
        return {}
    enrichment: dict[str, dict[str, Any]] = {}
    params = normalize_params({"analysisDepth": analysis_depth})
    params["persist_world_model"] = False
    for start in range(0, len(paths), max(1, batch_size)):
        batch_paths = paths[start : start + max(1, batch_size)]
        assets: list[UploadedImage] = []
        path_by_filename: dict[str, Path] = {}
        for path in batch_paths:
            try:
                assets.append(UploadedImage(path.name, path.read_bytes()))
                path_by_filename[path.name] = path
            except Exception:
                continue
        if not assets:
            continue
        try:
            analysis = analyze(assets, params)
        except Exception as exc:
            print(f"AI enrichment skipped for batch starting at {start}: {exc}")
            continue
        model_status = analysis.get("modelStatus", {})
        for entry in analysis.get("images", []) or []:
            filename = str(entry.get("filename", ""))
            path = path_by_filename.get(filename)
            if not path:
                continue
            emotions = [
                label
                for label, _score in sorted(
                    (entry.get("emotionScores") or {}).items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )
            ]
            affects = [
                str(observation.get("label", ""))
                for observation in entry.get("observations", []) or []
                if str(observation.get("type", "")) == "affect" and str(observation.get("label", "")).strip()
            ]
            observations = []
            for observation in entry.get("observations", []) or []:
                observations.append(
                    {
                        "label": observation.get("label"),
                        "type": observation.get("type"),
                        "confidence": observation.get("confidence"),
                        "source": observation.get("source"),
                        "bbox": observation.get("bbox"),
                        "metadata": observation.get("metadata", {}),
                    }
                )
            enrichment[str(path.resolve())] = {
                "caption": entry.get("caption"),
                "tags": unique_terms(entry.get("tags", [])),
                "objects": unique_terms(entry.get("objects", [])),
                "symbols": unique_terms(entry.get("symbols", [])),
                "textures": unique_terms(entry.get("textures", [])),
                "style_tags": unique_terms(entry.get("styles", [])),
                "emotion_tags": unique_terms(emotions),
                "affect_tags": unique_terms(affects or entry.get("affects", [])),
                "composition": unique_terms(entry.get("composition", [])),
                "observations": observations[:80],
                "scores": entry.get("scores", {}),
                "attention": entry.get("attention", {}),
                "zero_shot": entry.get("zeroShot", []),
                "analysis_version": analysis.get("modelStatus", {}).get("version"),
                "model_status": {
                    "embeddingModel": model_status.get("embeddingModel"),
                    "captionModel": model_status.get("captionModel"),
                    "groundingModel": model_status.get("groundingModel"),
                    "zeroShotModel": model_status.get("zeroShotModel"),
                    "affectiveModel": model_status.get("affectiveModel"),
                    "fusionModel": model_status.get("fusionModel"),
                },
            }
    return enrichment


def sample_from_image(
    path: Path,
    dataset_origin: str,
    safety_rating: str,
    license_policy: str,
    ai_enrichment: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        data = path.read_bytes()
        image = Image.open(path).convert("RGB")
    except Exception:
        return None

    digest = hashlib.sha256(data).hexdigest()
    palette = image_palette(image, 12)
    metrics = image_metrics(image)
    image.close()
    color_names = []
    for color in palette:
        name = str(color.get("name", "")).strip()
        if name and name not in color_names:
            color_names.append(name)

    ai_enrichment = ai_enrichment or {}
    composition = unique_terms([metrics["orientation"]] + list(ai_enrichment.get("composition", []) or []), limit=16)
    modalities = {
        "objects": unique_terms(ai_enrichment.get("objects", []) or []),
        "symbols": unique_terms(ai_enrichment.get("symbols", []) or []),
        "textures": unique_terms(ai_enrichment.get("textures", []) or []),
        "style_tags": unique_terms(ai_enrichment.get("style_tags", []) or []),
        "emotion_tags": unique_terms(ai_enrichment.get("emotion_tags", []) or []),
        "affect_tags": unique_terms(ai_enrichment.get("affect_tags", []) or []),
        "composition": composition,
        "colors": color_names,
        "tags": unique_terms(ai_enrichment.get("tags", []) or []),
    }

    return {
        "sample_id": f"{dataset_origin}:{digest[:16]}",
        "asset_ref": {
            "kind": "local_path",
            "value": relative_path(path),
            "redistribution": "local_only",
            "sha256": digest,
        },
        "dataset_origin": dataset_origin,
        "license_policy": license_policy,
        "safety": {
            "rating": safety_rating,
            "notes": "Local user-provided dataset sample; not redistributed.",
        },
        "raw_metadata": {
            "filename": path.name,
            "folder": relative_path(path.parent),
            "ingested": time.time(),
            **metrics,
        },
        "palette": palette,
        "modalities": modalities,
        "ai_enrichment": {
            "enabled": bool(ai_enrichment),
            "caption": ai_enrichment.get("caption"),
            "observations": ai_enrichment.get("observations", []),
            "scores": ai_enrichment.get("scores", {}),
            "attention": ai_enrichment.get("attention", {}),
            "zero_shot": ai_enrichment.get("zero_shot", []),
            "analysis_version": ai_enrichment.get("analysis_version"),
            "model_status": ai_enrichment.get("model_status", {}),
        },
        "embeddings": {
            "visual": [],
            "objects": [],
            "colors": [],
            "symbols": [],
            "textures": [],
            "styles": [],
            "emotions": [],
            "composition": [],
            "attention": (ai_enrichment.get("attention", {}) or {}).get("embedding", []),
            "unified_aesthetic": [],
        },
        "aesthetic_score": None,
        "neighbors": [],
        "relations": [
            {"source": "image", "target": label, "type": key, "weight": 1.0}
            for key, values in modalities.items()
            if key != "colors"
            for label in values[:24]
        ],
    }


def iter_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS)


def main_with_args(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Normalize local image folders into world-model JSONL samples.")
    parser.add_argument("folders", nargs="+", help="Local dataset folders to ingest.")
    parser.add_argument("--output", default=str(WORLD_SAMPLES_PATH), help="Output JSONL path.")
    parser.add_argument("--limit-per-folder", type=int, default=0, help="Optional max images per folder; 0 means all.")
    parser.add_argument("--safety-rating", default="safe", choices=["safe", "sensitive", "nsfw", "unknown"])
    parser.add_argument("--license-policy", default="local user dataset; metadata and embeddings only")
    parser.add_argument("--enrich-ai", action="store_true", help="Run local AI analysis once and persist semantic modalities in the JSONL.")
    parser.add_argument("--analysis-depth", default="fast", choices=sorted(VALID_ANALYSIS_DEPTHS), help="Analysis depth used by --enrich-ai.")
    parser.add_argument("--enrichment-batch-size", type=int, default=4, help="Images per offline AI enrichment batch.")
    args = parser.parse_args(argv)

    output = Path(args.output)
    if not output.is_absolute():
        output = BASE_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for folder_value in args.folders:
        folder = Path(folder_value)
        if not folder.is_absolute():
            folder = BASE_DIR / folder
        if not folder.exists() or not folder.is_dir():
            print(f"Skipping missing folder: {folder}")
            continue
        dataset_origin = folder.name
        paths = iter_images(folder)
        if args.limit_per_folder > 0:
            paths = paths[: args.limit_per_folder]
        enrichment_by_path = semantic_enrichment_for_paths(paths, args.analysis_depth, args.enrichment_batch_size) if args.enrich_ai else {}
        for path in paths:
            sample = sample_from_image(
                path,
                dataset_origin,
                args.safety_rating,
                args.license_policy,
                enrichment_by_path.get(str(path.resolve())),
            )
            if not sample:
                continue
            sample_id = str(sample["sample_id"])
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            samples.append(sample)

    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {output}")


def main() -> None:
    main_with_args()


if __name__ == "__main__":
    main()
