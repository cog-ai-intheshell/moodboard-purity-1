"""Moodboard aesthetic analysis pipeline.

This module owns the runtime analysis path used by the HTTP API and the AI
orchestrator. It intentionally keeps web-server concerns out: callers pass
already-uploaded images plus normalized parameters, and receive the analysis
payload consumed by Bento, Graph, PDF export and world-memory updates.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any

from PIL import Image

# Keep ML backends deterministic and safer under the stdlib threaded server.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

from src.moodboard.analysis.clustering import cluster_vectors
from src.moodboard.analysis.graph_builder import build_analysis_graph, canonical_color_key, cluster_color
from src.moodboard.analysis.purity import compute_latent_purity, finalize_purity, observation_counter
from src.moodboard.analysis.scoring import edge_density, heuristic_vector, infer_image_tags, normalize_scores
from src.moodboard.analysis.spectral import spectral_aesthetic_analysis
from src.moodboard.ai.models.affect.affect_retriever import apply_affective_model
from src.moodboard.ai.models.captions.concepts import caption_concepts, classify_terms_by_modality
from src.moodboard.ai.models.colors.palette_extractor import image_palette, palette_coherence_score
from src.moodboard.ai.models.detection.owlv2 import try_owlv2_grounding
from src.moodboard.ai.models.embeddings.siglip2 import try_siglip_embeddings_with_attention
from src.moodboard.ai.models.embeddings.zero_shot import apply_zero_shot_concepts
from src.moodboard.ai.models.world.aesthetic_index import match_aesthetics
from src.moodboard.ai.models.world.energy_model import update_world_model
from src.moodboard.ai.models.world.fusion_encoder import build_composite_vectors, mean_dense_vector
from src.moodboard.ai.models.world.mood_classifier import nearest_world_moods, nearest_world_samples
from src.moodboard.ai.registry import (
    ANALYSIS_MODEL_VERSION,
    MODEL_RECOMMENDATIONS,
    runtime_registry as model_runtime_registry,
    trained_artifacts_status,
)
from src.moodboard.ai.orchestrator import run_vision_language_tasks
from src.moodboard.ai.variables.labels import (
    add_observation,
    add_term_observations,
    clean_concept_label,
    make_observation,
    normalize_observation_type,
    rebuild_modalities_from_observations,
)
from src.moodboard.ai.variables.taxonomies import (
    build_world_taxonomy,
    load_aesthetic_knowledge,
)
from src.moodboard.core.cache import ANALYSIS_CACHE, ANALYSIS_RUNTIME_LOCK, analysis_cache_key, cleanup_analysis_cache
from src.moodboard.core.image_io import analyze_images, open_asset_rgb
from src.moodboard.core.paths import DATA_DIR
from src.moodboard.core.schemas import UploadedImage, clamp_float

os.environ.setdefault("HF_HOME", str(DATA_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

def apply_region_observations(
    image_entries: list[dict[str, Any]],
    region_observations: dict[int, list[dict[str, Any]]],
    source: str,
    term_counter: Counter[str],
) -> None:
    for idx, items in region_observations.items():
        if idx >= len(image_entries):
            continue
        entry = image_entries[idx]
        labels_for_classification = [str(item.get("label", "")).lower() for item in items if str(item.get("label", "")).strip()]
        region_classes = classify_terms_by_modality(labels_for_classification)
        for item in items:
            label = str(item.get("label", "")).lower()
            clean_label = clean_concept_label(label, max_words=4) or label
            classified = region_classes.get(clean_label, {})
            obs_type = normalize_observation_type(str(classified.get("type", "object")))
            if obs_type not in {"object", "symbol", "texture", "tag"}:
                obs_type = "object"
            add_observation(
                entry,
                make_observation(
                    clean_label,
                    obs_type,
                    max(float(item.get("confidence", 0.72)), float(classified.get("confidence", 0.0))),
                    source if source else classified.get("source", "region"),
                    bbox=item.get("bbox"),
                    metadata={"task": item.get("task", "region")},
                ),
            )
            term_counter.update([clean_label])
        rebuild_modalities_from_observations(entry)


def _analyze_moodboard_unlocked(assets: list[UploadedImage], params: dict[str, Any]) -> dict[str, Any]:
    cleanup_analysis_cache()
    key = analysis_cache_key(assets, params, ANALYSIS_MODEL_VERSION)
    cached = ANALYSIS_CACHE.get(key)
    if cached:
        payload = dict(cached["payload"])
        payload["cache"] = {"hit": True, "key": key[:12]}
        return payload

    infos = analyze_images(assets, params["orientation_threshold"])
    if not infos:
        raise ValueError("No usable images were found.")

    image_entries: list[dict[str, Any]] = []
    pil_for_ml: list[Image.Image] = []
    heuristic_vectors: list[list[float]] = []
    all_palette: list[dict[str, Any]] = []
    emotion_counter: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()
    density_values: list[float] = []

    for idx, info in enumerate(infos, 1):
        image = open_asset_rgb(info.asset)
        palette = image_palette(image, 12)
        density = edge_density(image)
        density_values.append(density)
        inferred = infer_image_tags(info, palette, density)
        vector = heuristic_vector(info, palette, density)
        heuristic_vectors.append(vector)
        pil_for_ml.append(image.copy())
        all_palette.extend(palette)
        emotion_scores = normalize_scores({emotion: 1.0 for emotion in inferred["emotions"]})
        emotion_counter.update(inferred["emotions"])
        for key_name in ("tags", "symbols", "styles", "composition"):
            term_counter.update(inferred[key_name])
        entry = {
            "id": f"image-{idx}",
            "filename": info.asset.filename,
            "orientation": info.orientation,
            "width": info.width,
            "height": info.height,
            "palette": palette,
            "tags": inferred["tags"],
            "symbols": inferred["symbols"],
            "objects": [],
            "textures": [],
            "styles": inferred["styles"],
            "affects": [],
            "composition": inferred["composition"],
            "emotionScores": emotion_scores,
            "observations": [],
            "scores": {
                "brightness": round(info.brightness, 4),
                "contrast": round(info.contrast, 4),
                "saturation": round(info.hsv[1], 4),
                "density": round(density, 4),
                "hero": round(clamp_float(info.hero_score / 4.6, 0.0, 0.0, 1.0), 4),
                "silhouetteStrength": round(clamp_float((density * 0.52) + (info.contrast * 0.38) + (abs(info.hsv[2] - 0.5) * 0.20), 0.0, 0.0, 1.0), 4),
            },
            "cluster": 0,
            "attention": {"embedding": [], "stats": {"source": "unavailable"}},
            "outlier": False,
        }
        add_term_observations(entry, inferred["tags"], "tag", "heuristic-visual-fallback", 0.38)
        add_term_observations(entry, inferred["symbols"], "symbol", "heuristic-visual-fallback", 0.34)
        add_term_observations(entry, inferred["styles"], "style", "heuristic-visual-fallback", 0.36)
        add_term_observations(entry, inferred["composition"], "composition", "heuristic-visual-fallback", 0.42)
        add_term_observations(entry, inferred["emotions"], "emotion", "heuristic-visual-fallback", 0.32)
        for color in palette:
            add_observation(
                entry,
                make_observation(
                    color["name"],
                    "color",
                    max(0.08, float(color.get("weight", 0.0))),
                    "lab-mediancut",
                    metadata={"hex": color["hex"], "rgb": color["rgb"], "role": color.get("role", "accent")},
                ),
            )
        image_entries.append(entry)
        image.close()

    ml_vectors, attention_signatures, model_status = try_siglip_embeddings_with_attention(pil_for_ml)
    for idx, signature in enumerate(attention_signatures):
        if idx < len(image_entries):
            image_entries[idx]["attention"] = signature
    captions, region_observations, caption_status = run_vision_language_tasks(pil_for_ml, params.get("analysis_depth", "balanced"))
    vectors = ml_vectors or heuristic_vectors

    zero_shot_status = apply_zero_shot_concepts(image_entries, vectors, emotion_counter, term_counter) if ml_vectors else "heuristic-only"

    for idx, caption in captions.items():
        if idx >= len(image_entries):
            continue
        entry = image_entries[idx]
        entry["caption"] = caption
        enrichment = caption_concepts(caption)
        current_emotions = set(entry.get("emotionScores", {}).keys())
        for emotion in enrichment["emotions"]:
            if emotion not in current_emotions:
                emotion_counter.update([emotion])
                current_emotions.add(emotion)
        if current_emotions:
            entry["emotionScores"] = normalize_scores({emotion: 1.0 for emotion in current_emotions})
        for key_name in ("tags", "objects", "symbols", "textures", "styles", "composition"):
            existing = set(entry.get(key_name, []))
            added = [term for term in enrichment[key_name] if term not in existing]
            if added:
                term_counter.update(added)
                entry[key_name] = sorted(existing | set(added))
            obs_type = normalize_observation_type(key_name)
            add_term_observations(entry, enrichment[key_name], obs_type, "florence-caption", 0.68)
        add_term_observations(entry, enrichment["emotions"], "emotion", "florence-caption", 0.66)
        rebuild_modalities_from_observations(entry)

    apply_region_observations(image_entries, region_observations, "florence-region", term_counter)
    owlv2_taxonomy = build_world_taxonomy(load_aesthetic_knowledge(), max_per_type=120)
    owlv2_regions, grounding_status = try_owlv2_grounding(
        pil_for_ml,
        image_entries,
        owlv2_taxonomy,
        params.get("analysis_depth", "balanced"),
    )
    apply_region_observations(image_entries, owlv2_regions, "owlv2-open-vocabulary", term_counter)
    for image in pil_for_ml:
        image.close()

    palette_counts: dict[str, dict[str, Any]] = {}
    for color in all_palette:
        key_name = canonical_color_key(str(color.get("name", "")))
        weight = float(color.get("weight", 0.0) or 0.0)
        if key_name not in palette_counts:
            palette_counts[key_name] = {
                "name": color["name"],
                "hex": color["hex"],
                "rgb": color["rgb"],
                "weight": 0.0,
                "_rgb_weight": [0.0, 0.0, 0.0],
                "_dominant_weight": 0.0,
            }
        palette_counts[key_name]["weight"] += weight
        for channel_idx, channel in enumerate(color.get("rgb", [0, 0, 0])[:3]):
            palette_counts[key_name]["_rgb_weight"][channel_idx] += float(channel) * weight
        if weight > float(palette_counts[key_name].get("_dominant_weight", 0.0)):
            palette_counts[key_name]["hex"] = color["hex"]
            palette_counts[key_name]["_dominant_weight"] = weight
    global_palette = sorted(palette_counts.values(), key=lambda item: item["weight"], reverse=True)
    total_palette_weight = max(1e-6, sum(item["weight"] for item in global_palette))
    for item in global_palette:
        raw_weight = max(1e-6, float(item.get("weight", 0.0)))
        averaged_rgb = [int(round(float(value) / raw_weight)) for value in item.pop("_rgb_weight", [0.0, 0.0, 0.0])]
        item["rgb"] = [max(0, min(255, value)) for value in averaged_rgb[:3]]
        item.pop("_dominant_weight", None)
        item["weight"] = round(item["weight"] / total_palette_weight, 4)

    term_counter = observation_counter(image_entries, {"tag", "symbol", "object", "texture", "style", "affect", "composition"})
    preliminary_global_vector = mean_dense_vector(vectors) if ml_vectors else None
    preliminary_aesthetic_matches = match_aesthetics(image_entries, term_counter, preliminary_global_vector)
    affective_status = (
        apply_affective_model(image_entries, vectors, load_aesthetic_knowledge(), preliminary_aesthetic_matches)
        if ml_vectors
        else "heuristic-affect-fallback"
    )
    if ml_vectors:
        emotion_counter = observation_counter(image_entries, {"emotion"})
    term_counter = observation_counter(image_entries, {"tag", "symbol", "object", "texture", "style", "affect", "composition", "emotion"})
    analysis_vectors, fusion_status = build_composite_vectors(image_entries, vectors)
    vectors = analysis_vectors

    labels, outlier_indices = cluster_vectors(vectors)
    for idx, label in enumerate(labels):
        image_entries[idx]["cluster"] = label
        image_entries[idx]["outlier"] = idx in outlier_indices

    global_vector = mean_dense_vector(vectors) if ml_vectors else None
    nearest_samples = nearest_world_samples(global_vector, 8)
    trained_mood_matches = nearest_world_moods(global_vector, 5)
    aesthetic_matches = match_aesthetics(image_entries, term_counter, global_vector)
    emotion_scores = normalize_scores({emotion: float(amount) for emotion, amount in emotion_counter.items()})
    dominant = aesthetic_matches[0] if aesthetic_matches else {"name": "Unknown", "score": 0.0}
    color_coherence = palette_coherence_score(global_palette)
    purity_metrics = compute_latent_purity(
        image_entries=image_entries,
        vectors=vectors,
        labels=labels,
        outlier_indices=outlier_indices,
        emotion_scores=emotion_scores,
        density_values=density_values,
        aesthetic_matches=aesthetic_matches,
        color_coherence=color_coherence,
    )
    cluster_counts = purity_metrics["clusterCounts"]
    cluster_total = purity_metrics["clusterTotal"]
    cluster_cohesion = purity_metrics["clusterCohesion"]
    cluster_separation = purity_metrics["clusterSeparation"]
    modality_coverage = purity_metrics["modalityCoverage"]
    aesthetic_margin = purity_metrics["aestheticMargin"]
    color_coherence = purity_metrics["colorCoherence"]
    style_coherence = purity_metrics["styleCoherence"]
    affect_coherence = purity_metrics["affectCoherence"]
    symbolic_coherence = purity_metrics["symbolicCoherence"]
    emotional_coherence = purity_metrics["emotionalCoherence"]
    composition_coherence = purity_metrics["compositionCoherence"]
    latent_purity = purity_metrics["latentPurity"]
    purity = purity_metrics["purity"]
    hybridation = purity_metrics["hybridation"]

    clusters = []
    for label, amount in sorted(cluster_counts.items()):
        members = [entry for entry in image_entries if entry["cluster"] == label]
        concepts = Counter(
            term
            for entry in members
            for term in entry["tags"] + entry.get("objects", []) + entry["symbols"] + entry.get("textures", []) + entry["styles"] + entry.get("affects", [])
        )
        clusters.append(
            {
                "id": int(label),
                "label": f"Cluster {label + 1}",
                "color": cluster_color(int(label)),
                "size": amount,
                "share": round(amount / cluster_total, 4),
                "cohesion": round(cluster_cohesion, 4),
                "separation": round(cluster_separation, 4),
                "palette": [name for name, _ in Counter(color["name"] for entry in members for color in entry.get("palette", [])).most_common(6)],
                "emotions": [name for name, _ in Counter(emotion for entry in members for emotion in entry.get("emotionScores", {}).keys()).most_common(6)],
                "objects": [term for term, _ in Counter(term for entry in members for term in entry.get("objects", [])).most_common(6)],
                "symbols": [term for term, _ in Counter(term for entry in members for term in entry.get("symbols", [])).most_common(6)],
                "textures": [term for term, _ in Counter(term for entry in members for term in entry.get("textures", [])).most_common(6)],
                "styles": [term for term, _ in Counter(term for entry in members for term in entry.get("styles", [])).most_common(6)],
                "affects": [term for term, _ in Counter(term for entry in members for term in entry.get("affects", [])).most_common(6)],
                "concepts": [term for term, _ in concepts.most_common(6)],
                "images": [entry["id"] for entry in members],
            }
        )

    graph = build_analysis_graph(image_entries, vectors, global_palette, emotion_scores, aesthetic_matches)
    spectral_analysis = spectral_aesthetic_analysis(graph, str(dominant.get("name", "Unknown")), latent_purity)
    final_purity_metrics = finalize_purity(
        latent_purity=latent_purity,
        spectral_analysis=spectral_analysis,
        trained_mood_matches=trained_mood_matches,
        aesthetic_margin=aesthetic_margin,
        modality_coverage=modality_coverage,
    )
    purity = final_purity_metrics["purity"]
    hybridation = final_purity_metrics["hybridation"]
    spectral_purity = final_purity_metrics["spectralPurity"]
    harmonicity_score = final_purity_metrics["harmonicity"]
    spectral_analysis = final_purity_metrics["spectralAnalysis"]
    genome_counter = observation_counter(image_entries, {"tag", "symbol", "object", "texture", "style", "affect", "composition", "emotion"})
    payload = {
        "modelStatus": {
            "embeddingModel": model_status,
            "fusionModel": fusion_status,
            "trainedArtifacts": trained_artifacts_status(),
            "captionModel": caption_status,
            "groundingModel": grounding_status,
            "zeroShotModel": zero_shot_status,
            "affectiveModel": affective_status,
            "analysisDepth": params.get("analysis_depth", "balanced"),
            "captionBackend": os.environ.get("MOODBOARD_CAPTION_BACKEND", "florence").lower(),
            "version": ANALYSIS_MODEL_VERSION,
            "runtimeRegistry": model_runtime_registry(),
            "recommendations": MODEL_RECOMMENDATIONS,
        },
        "images": image_entries,
        "globalProfile": {
            "dominantAesthetic": dominant,
            "secondaryAesthetics": aesthetic_matches[1:5],
            "nearestWorldSamples": nearest_samples,
            "trainedMoodMatches": trained_mood_matches,
            "moodSummary": ", ".join([term.title() for term, _ in emotion_counter.most_common(3)]) or "Balanced moodboard",
            "aestheticGenome": {
                term: round(amount / max(1, sum(genome_counter.values())), 4)
                for term, amount in genome_counter.most_common(18)
            },
        },
        "palette": global_palette[:12],
        "scores": {
            "purity": round(purity, 4),
            "purityScore": round(purity, 4),
            "latentPurity": round(latent_purity, 4),
            "spectralPurity": round(spectral_purity, 4),
            "harmonicity": round(harmonicity_score, 4),
            "harmonyCoherence": round(harmonicity_score, 4),
            "hybridation": round(hybridation, 4),
            "clusterCohesion": round(cluster_cohesion, 4),
            "clusterSeparation": round(cluster_separation, 4),
            "modalityCoverage": round(modality_coverage, 4),
            "colorCoherence": round(color_coherence, 4),
            "styleCoherence": round(style_coherence, 4),
            "affectCoherence": round(affect_coherence, 4),
            "symbolicCoherence": round(symbolic_coherence, 4),
            "emotionalCoherence": round(emotional_coherence, 4),
            "compositionCoherence": round(composition_coherence, 4),
        },
        "clusters": clusters,
        "outliers": [image_entries[idx] for idx in outlier_indices],
        "aestheticMatches": aesthetic_matches,
        "graph": graph,
        "spectralAnalysis": spectral_analysis,
        "cache": {"hit": False, "key": key[:12]},
    }
    if params.get("persist_world_model", True):
        payload["worldModel"] = update_world_model(payload, key)
    else:
        payload["worldModel"] = {"updated": False, "policy": "offline-analysis-no-runtime-persistence"}
    ANALYSIS_CACHE[key] = {"created": time.time(), "payload": payload}
    return payload



def analyze_moodboard(assets: list[UploadedImage], params: dict[str, Any]) -> dict[str, Any]:
    """Run the full analysis pipeline under the shared ML runtime lock."""

    # UMAP/Numba and optional ML runtimes can crash or corrupt state when two
    # request threads enter kernels at once. The lock belongs to core/cache so
    # every entrypoint shares the same protection.
    with ANALYSIS_RUNTIME_LOCK:
        return _analyze_moodboard_unlocked(assets, params)


def analyze(files: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias used by older orchestration code."""

    return analyze_moodboard(files, params)
