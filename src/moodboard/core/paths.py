"""Filesystem locations shared by the app, training jobs and scrapers.

This module is intentionally tiny and dependency-free: every other package can
import it without pulling the web server or any ML model into memory.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_STATIC_DIR = FRONTEND_DIR / "static"
HTML_PATH = FRONTEND_DIR / "moodboard_interface.html"

DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"

AESTHETICS_CACHE_PATH = DATABASE_DIR / "aesthetics_cache.json"
WORLD_MODEL_PATH = DATABASE_DIR / "world_model_index.json"
WORLD_SAMPLES_PATH = DATABASE_DIR / "world_samples" / "local_moodboards.jsonl"
DATASET_REGISTRY_PATH = DATABASE_DIR / "dataset_registry.json"
MODEL_REGISTRY_PATH = DATABASE_DIR / "model_registry.json"
COLOR_NAMES_PATH = DATABASE_DIR / "color_names.json"

TRAINED_MODELS_DIR = DATA_DIR / "trained_models"
AESTHETIC_INDEX_PATH = TRAINED_MODELS_DIR / "aesthetic_text_index_v1.json"
FUSION_CALIBRATOR_PATH = TRAINED_MODELS_DIR / "fusion_calibrator_v1.json"
WORLD_SAMPLE_INDEX_PATH = TRAINED_MODELS_DIR / "world_sample_index_v1.json"
WORLD_MOOD_CLASSIFIER_PATH = TRAINED_MODELS_DIR / "world_mood_classifier_v1.json"
