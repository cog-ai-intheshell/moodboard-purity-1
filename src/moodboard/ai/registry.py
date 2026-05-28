"""Central runtime registry for local AI models and trained artifacts.

The API layer asks this module what is installed, active and recommended. It is
kept independent from the web server so model orchestration can evolve without
pulling request handling or heavy model weights into memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.moodboard.core.paths import (
    AESTHETIC_INDEX_PATH,
    BASE_DIR,
    DATASET_REGISTRY_PATH,
    DATA_DIR,
    FUSION_CALIBRATOR_PATH,
    WORLD_MOOD_CLASSIFIER_PATH,
    WORLD_SAMPLE_INDEX_PATH,
)


ANALYSIS_MODEL_VERSION = "mvp-lite-world-model-harmony-v23"
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
FAST_CAPTION_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
QUALITY_VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OPEN_VOCAB_DETECTOR_MODEL_ID = "google/owlv2-base-patch16-ensemble"
DINO_MODEL_ID = "facebook/dinov2-base"
SAM_MODEL_ID = "facebook/sam2-hiera-small"
FLORENCE_CAPTION_LIMIT = 24

MODEL_RECOMMENDATIONS: dict[str, Any] = {
    "multimodal_embeddings": {
        "active": SIGLIP_MODEL_ID,
        "upgrade": "google/siglip2-so400m-patch14-384",
        "role": "image/text embeddings, zero-shot concepts, aesthetic retrieval",
    },
    "caption_objects_symbols": {
        "active": FLORENCE_MODEL_ID,
        "upgrade": "microsoft/Florence-2-large-ft",
        "role": "captions, dense region captions, object detection, OCR",
    },
    "fast_captioning": {
        "recommended": FAST_CAPTION_MODEL_ID,
        "qualityAlternate": QUALITY_VLM_MODEL_ID,
        "tinyAlternate": "vikhyatk/moondream2",
        "role": "faster captions for fast/balanced analysis; keep Florence-2 for object boxes and dense regions",
    },
    "emotion_affect": {
        "active": "SigLIP2 text-image similarity over the local aesthetic corpus vocabulary",
        "upgrade": "fine-tuned affect/aesthetic head on a curated moodboard dataset",
        "role": "latent affect retrieval instead of fixed hand-written mood categories",
    },
    "modality_fusion": {
        "active": "weighted local fusion of visual, color, object, symbol, texture, style, affect and composition embeddings",
        "upgrade": "train a small fusion encoder with contrastive + graph-neighborhood objectives",
        "role": "unified aesthetic embedding used by clustering, graph layout, purity and spectral analysis",
    },
    "open_vocabulary_detection": {
        "recommended": OPEN_VOCAB_DETECTOR_MODEL_ID,
        "alternate": "IDEA-Research/grounding-dino-base",
        "role": "local open-vocabulary object and symbol grounding",
    },
    "visual_backbone": {
        "recommended": DINO_MODEL_ID,
        "role": "style/silhouette/visual retrieval features beyond language alignment",
    },
    "attention_embedding": {
        "active": "SigLIP2 patch-token salience signature",
        "upgrade": "ViT attention rollout from an eager-attention vision transformer",
        "role": "image attention/salience similarity, inspired by ViT attention-distance analysis",
    },
    "segmentation": {
        "recommended": SAM_MODEL_ID,
        "role": "silhouette, masks, foreground geometry, object regions",
    },
    "clustering": {
        "active": "umap-learn + hdbscan",
        "role": "non-parametric latent regimes with outlier support",
    },
    "training_corpora": {
        "registry": str(DATASET_REGISTRY_PATH.relative_to(BASE_DIR)),
        "role": "common schema and dataset readiness map for the long-term world model",
    },
}


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception:
        return None
    return None


def trained_artifacts_status() -> dict[str, Any]:
    """Return status for model artifacts produced offline by world_learning."""

    def artifact(path: Path) -> dict[str, Any]:
        payload = _read_json_file(path)
        return {
            "path": str(path.relative_to(BASE_DIR)),
            "available": payload is not None,
            "updated": payload.get("updated") if payload else None,
            "version": payload.get("version") if payload else None,
            "itemCount": len(payload.get("items", [])) if payload and isinstance(payload.get("items"), list) else None,
        }

    return {
        "policy": "Artifacts are produced by world_learning jobs and loaded by the app; the app never trains on startup.",
        "aestheticIndex": artifact(AESTHETIC_INDEX_PATH),
        "fusionCalibrator": artifact(FUSION_CALIBRATOR_PATH),
        "worldSampleIndex": artifact(WORLD_SAMPLE_INDEX_PATH),
        "worldMoodClassifier": artifact(WORLD_MOOD_CLASSIFIER_PATH),
    }


def hf_cache_dir_for_model(model_id: str) -> Path:
    """Resolve the expected local Hugging Face cache folder for a model id."""

    return DATA_DIR / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))


def local_model_status(model_id: str) -> dict[str, Any]:
    """Inspect whether a configured Hugging Face model is present locally."""

    cache_dir = hf_cache_dir_for_model(model_id)
    refs_main = cache_dir / "refs" / "main"
    snapshots_dir = cache_dir / "snapshots"
    snapshot_count = 0
    if snapshots_dir.exists():
        try:
            snapshot_count = sum(1 for child in snapshots_dir.iterdir() if child.is_dir())
        except OSError:
            snapshot_count = 0
    return {
        "id": model_id,
        "available": bool(cache_dir.exists() and snapshot_count > 0),
        "cachePath": str(cache_dir.relative_to(BASE_DIR)) if cache_dir.exists() else None,
        "snapshotCount": snapshot_count,
        "revision": refs_main.read_text(encoding="utf-8").strip() if refs_main.exists() else None,
    }


def runtime_registry() -> dict[str, Any]:
    """Return the active local model registry consumed by `/api/models`."""

    models = {
        "multimodal_embeddings": {
            **local_model_status(SIGLIP_MODEL_ID),
            "active": True,
            "requiredFor": ["image embeddings", "text embeddings", "zero-shot concepts", "aesthetic matching", "fusion"],
        },
        "caption_regions": {
            **local_model_status(FLORENCE_MODEL_ID),
            "active": os.environ.get("MOODBOARD_ENABLE_FLORENCE", "1").lower() not in {"0", "false", "no"},
            "requiredFor": ["captions", "object boxes", "dense region captions"],
        },
        "fast_captioning": {
            **local_model_status(FAST_CAPTION_MODEL_ID),
            "active": os.environ.get("MOODBOARD_CAPTION_BACKEND", "florence").lower() in {"auto", "fast", "smolvlm", "smolvlm2"},
            "requiredFor": ["fast captions"],
        },
        "quality_vlm": {
            **local_model_status(QUALITY_VLM_MODEL_ID),
            "active": False,
            "requiredFor": ["future deeper semantic captions", "visual reasoning"],
        },
        "open_vocabulary_detection": {
            **local_model_status(OPEN_VOCAB_DETECTOR_MODEL_ID),
            "active": os.environ.get("MOODBOARD_ENABLE_OWLV2", "1").lower() not in {"0", "false", "no"},
            "requiredFor": ["future local symbol/object grounding"],
        },
        "visual_backbone": {
            **local_model_status(DINO_MODEL_ID),
            "active": False,
            "requiredFor": ["future style and silhouette embeddings"],
        },
        "attention_embedding": {
            **local_model_status(SIGLIP_MODEL_ID),
            "active": True,
            "requiredFor": ["patch-token salience signatures", "attention similarity edges"],
        },
        "segmentation": {
            **local_model_status(SAM_MODEL_ID),
            "active": False,
            "requiredFor": ["future masks and silhouette geometry"],
        },
    }
    return {
        "version": ANALYSIS_MODEL_VERSION,
        "offline": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
        "captionBackend": os.environ.get("MOODBOARD_CAPTION_BACKEND", "florence").lower(),
        "trainedArtifacts": trained_artifacts_status(),
        "models": models,
        "recommendations": MODEL_RECOMMENDATIONS,
    }
