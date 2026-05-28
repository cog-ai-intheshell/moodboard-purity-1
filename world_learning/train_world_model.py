#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train/precompute local world-model artifacts once.

This script is intentionally separate from the web app. The app loads artifacts
from data/trained_models and never trains them at startup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.moodboard.ai.models.world.aesthetic_index import build_aesthetic_embedding_index, read_json_file
from src.moodboard.ai.models.world.fusion_encoder import MODALITY_FUSION_WEIGHTS
from src.moodboard.ai.models.world.mood_classifier import build_world_mood_classifier
from src.moodboard.ai.models.world.sample_index import build_world_sample_index
from src.moodboard.ai.registry import ANALYSIS_MODEL_VERSION
from src.moodboard.core.paths import (
    AESTHETIC_INDEX_PATH,
    BASE_DIR,
    FUSION_CALIBRATOR_PATH,
    TRAINED_MODELS_DIR,
    WORLD_MOOD_CLASSIFIER_PATH,
    WORLD_SAMPLE_INDEX_PATH,
    WORLD_SAMPLES_PATH,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def calibrate_fusion_weights(classifier_payload: dict | None = None) -> dict:
    # Keep the modality weights as a stable prior, but persist the supervised
    # world-classifier diagnostics next to them so the runtime can report which
    # training data actually informed this pass.
    weights = dict(MODALITY_FUSION_WEIGHTS)
    total = sum(max(0.0, value) for value in weights.values()) or 1.0
    weights = {key: round(value / total, 6) for key, value in weights.items()}
    classifier_metrics = (classifier_payload or {}).get("metrics", {}) if isinstance(classifier_payload, dict) else {}
    class_count = int(classifier_metrics.get("classCount", 0) or 0)
    sample_count = int((classifier_payload or {}).get("sampleCount", 0) or 0) if isinstance(classifier_payload, dict) else 0
    return {
        "version": "fusion-calibrator-v1",
        "analysisModelVersion": ANALYSIS_MODEL_VERSION,
        "updated": time.time(),
        "method": "stable expert-prior weights plus supervised local world-classifier diagnostics",
        "weights": weights,
        "trainingData": {
            "worldMoodClassifier": str(WORLD_MOOD_CLASSIFIER_PATH.relative_to(BASE_DIR)),
            "classCount": class_count,
            "sampleCount": sample_count,
            "separation": classifier_metrics.get("separation"),
            "meanIntraSimilarity": classifier_metrics.get("meanIntraSimilarity"),
            "meanInterSimilarity": classifier_metrics.get("meanInterSimilarity"),
        },
        "notes": [
            "Loaded by the app at analysis time.",
            "The app does not retrain this artifact at startup.",
            "Fusion weights stay conservative until large external datasets provide enough labeled modalities.",
            "The mood classifier is trained once from the normalized sample index and loaded directly by the app."
        ]
    }


def main_with_args(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/precompute local aesthetic world-model artifacts.")
    parser.add_argument("--skip-aesthetic-index", action="store_true", help="Do not rebuild the SigLIP2 aesthetic text index.")
    parser.add_argument("--skip-fusion", action="store_true", help="Do not rewrite the fusion calibrator artifact.")
    parser.add_argument("--skip-sample-index", action="store_true", help="Do not rebuild the local world sample embedding index.")
    parser.add_argument("--skip-world-classifier", action="store_true", help="Do not rebuild the supervised world mood classifier.")
    parser.add_argument("--samples-jsonl", default=str(WORLD_SAMPLES_PATH), help="World-model samples JSONL produced by world_learning/data_factory.py.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional max samples to index; 0 means all.")
    parser.add_argument("--batch-size", type=int, default=64, help="Text embedding batch size.")
    parser.add_argument("--sample-batch-size", type=int, default=8, help="Image embedding batch size for samples.")
    args = parser.parse_args(argv)

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_aesthetic_index:
        index_payload = build_aesthetic_embedding_index(batch_size=args.batch_size)
        write_json(AESTHETIC_INDEX_PATH, index_payload)
        print(f"Wrote {AESTHETIC_INDEX_PATH} ({index_payload['itemCount']} aesthetics)")

    samples_path = Path(args.samples_jsonl)
    if not samples_path.is_absolute():
        samples_path = BASE_DIR / samples_path
    sample_payload = None
    if not args.skip_sample_index:
        if samples_path.exists():
            sample_payload = build_world_sample_index(
                samples_path=samples_path,
                batch_size=args.sample_batch_size,
                max_items=args.max_samples or None,
            )
            write_json(WORLD_SAMPLE_INDEX_PATH, sample_payload)
            print(f"Wrote {WORLD_SAMPLE_INDEX_PATH} ({sample_payload['itemCount']} samples)")
        else:
            print(f"Skipping sample index; samples JSONL not found: {samples_path}")
    else:
        sample_payload = read_json_file(WORLD_SAMPLE_INDEX_PATH)

    classifier_payload = None
    if not args.skip_world_classifier:
        if WORLD_SAMPLE_INDEX_PATH.exists():
            classifier_payload = build_world_mood_classifier(WORLD_SAMPLE_INDEX_PATH)
            write_json(WORLD_MOOD_CLASSIFIER_PATH, classifier_payload)
            print(f"Wrote {WORLD_MOOD_CLASSIFIER_PATH} ({classifier_payload['itemCount']} classes)")
        else:
            print(f"Skipping world classifier; sample index not found: {WORLD_SAMPLE_INDEX_PATH}")
    else:
        classifier_payload = read_json_file(WORLD_MOOD_CLASSIFIER_PATH)

    if not args.skip_fusion:
        fusion_payload = calibrate_fusion_weights(classifier_payload=classifier_payload)
        write_json(FUSION_CALIBRATOR_PATH, fusion_payload)
        print(f"Wrote {FUSION_CALIBRATOR_PATH}")

    print("Training artifacts ready. The web app will load them directly.")


def main() -> None:
    main_with_args()


if __name__ == "__main__":
    main()
