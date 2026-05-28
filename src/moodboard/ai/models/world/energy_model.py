"""Compatibility-energy and local world-memory helpers.

This module is the runtime boundary for the future predictive world model. The
current MVP still uses a lightweight JSON memory, but the analyzer should not
know how that memory is stored or compared.
"""

from __future__ import annotations

import json
import math
import sys
import time
from typing import Any

from src.moodboard.ai.registry import ANALYSIS_MODEL_VERSION, SIGLIP_MODEL_ID
from src.moodboard.core.paths import DATABASE_DIR, WORLD_MODEL_PATH


WORLD_SAMPLE_SCHEMA_VERSION = "aesthetic-world-sample-v1"


def energy_from_purity(purity: float) -> float:
    """Convert a purity score into a simple compatibility energy.

    Lower energy means the moodboard is more internally compatible. This is not
    yet the final learned EBM/JEPA energy; it is a stable placeholder boundary.
    """

    return max(0.0, min(1.0, 1.0 - purity))


def sparse_profile_vector(summary: dict[str, Any]) -> dict[str, float]:
    """Represent a moodboard summary as a sparse concept/color/aesthetic vector."""

    vector: dict[str, float] = {}
    for key, amount in (summary.get("genome") or {}).items():
        vector[f"concept:{key}"] = float(amount)
    for item in summary.get("palette", []) or []:
        vector[f"color:{item.get('name')}"] = float(item.get("weight", 0.0))
    for idx, item in enumerate(summary.get("aesthetics", []) or []):
        vector[f"aesthetic:{item.get('name')}"] = max(0.0, float(item.get("score", 0.0)) * (1.0 - idx * 0.08))
    return vector


def sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosine similarity for sparse world-memory profiles."""

    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return dot / (left_norm * right_norm)


def update_world_model(payload: dict[str, Any], analysis_key: str) -> dict[str, Any]:
    """Persist a compact moodboard memory record and return nearest records."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    record_id = analysis_key[:16]
    profile = payload.get("globalProfile", {})
    modality_counts: dict[str, int] = {}
    for node in (payload.get("graph") or {}).get("nodes", []) or []:
        node_type = str(node.get("type", ""))
        modality_counts[node_type] = modality_counts.get(node_type, 0) + 1

    summary = {
        "schemaVersion": WORLD_SAMPLE_SCHEMA_VERSION,
        "id": record_id,
        "created": time.time(),
        "imageCount": len(payload.get("images", [])),
        "dominantAesthetic": profile.get("dominantAesthetic", {}),
        "aesthetics": payload.get("aestheticMatches", [])[:8],
        "genome": profile.get("aestheticGenome", {}),
        "palette": payload.get("palette", [])[:12],
        "scores": payload.get("scores", {}),
        "modalities": modality_counts,
        "embeddingPolicy": {
            "visual": SIGLIP_MODEL_ID,
            "fusion": "weighted modality text embeddings + visual embedding",
            "schema": WORLD_SAMPLE_SCHEMA_VERSION,
        },
        "clusters": [
            {
                "id": cluster.get("id"),
                "share": cluster.get("share"),
                "concepts": cluster.get("concepts", [])[:8],
            }
            for cluster in payload.get("clusters", [])
        ],
    }

    index: dict[str, Any] = {"version": 1, "records": {}}
    if WORLD_MODEL_PATH.exists():
        try:
            loaded = json.loads(WORLD_MODEL_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                index.update(loaded)
        except Exception as exc:
            print(f"[WARN] Cannot read world model index: {exc}", file=sys.stderr)

    records = index.setdefault("records", {})
    current_vector = sparse_profile_vector(summary)
    nearest = []
    for other_id, other in records.items():
        if other_id == record_id or not isinstance(other, dict):
            continue
        score = sparse_cosine(current_vector, sparse_profile_vector(other))
        nearest.append(
            {
                "id": other_id,
                "similarity": round(score, 4),
                "dominantAesthetic": (other.get("dominantAesthetic") or {}).get("name", "Unknown"),
                "purity": (other.get("scores") or {}).get("purity"),
            }
        )
    nearest.sort(key=lambda item: item["similarity"], reverse=True)

    records[record_id] = summary
    index["updated"] = time.time()
    index["modelVersion"] = ANALYSIS_MODEL_VERSION
    WORLD_MODEL_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "recordId": record_id,
        "recordCount": len(records),
        "nearestMoodboards": nearest[:5],
        "status": "local-json-world-model",
        "next": "Use this index as the seed for FAISS/ChromaDB + graph persistence when the corpus grows.",
    }
