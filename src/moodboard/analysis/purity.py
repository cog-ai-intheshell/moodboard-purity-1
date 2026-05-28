"""Moodboard purity and coherence scoring.

The analyzer builds many intermediate signals: latent image vectors, graph
clusters, palette coherence, detected concepts and spectral metrics. This
module keeps the score aggregation in one place so the API, PDF export and UI
can all read the same purity values instead of recomputing near-duplicates.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .scoring import cosine_similarity


def clamp_float(value: Any, default: float, low: float, high: float) -> float:
    """Parse and clamp a numeric value while tolerating missing model outputs."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def normalize(value: float) -> float:
    """Compatibility helper for older callers that only need a 0..1 clamp."""

    return max(0.0, min(1.0, value))


def observation_counter(image_entries: list[dict[str, Any]], obs_types: set[str] | None = None) -> Counter[str]:
    """Count normalized observations, weighted by their confidence.

    Observations are the common currency for object, symbol, affect, style and
    color signals. Weighting them here makes the purity score follow model
    confidence without hard-coding modality labels into the UI.
    """

    counter: Counter[str] = Counter()
    for entry in image_entries:
        for observation in entry.get("observations", []):
            obs_type = str(observation.get("type", ""))
            if obs_types and obs_type not in obs_types:
                continue
            label = str(observation.get("label", "")).lower()
            if label:
                counter[label] += max(1, int(round(float(observation.get("confidence", 0.0)) * 4)))
    return counter


def _top_share(counter: Counter[str], top_n: int, default: float) -> float:
    """Return the share carried by the strongest concepts in a modality."""

    total = sum(counter.values())
    if total <= 0:
        return default
    return clamp_float(sum(amount for _, amount in counter.most_common(top_n)) / total, default, 0.0, 1.0)


def _aesthetic_margin(aesthetic_matches: list[dict[str, Any]]) -> float:
    """Score how clearly one aesthetic dominates the following candidates."""

    dominant = float(aesthetic_matches[0].get("score", 0.0) or 0.0) if aesthetic_matches else 0.0
    secondary = float(aesthetic_matches[1].get("score", 0.0) or 0.0) if len(aesthetic_matches) > 1 else 0.0
    return clamp_float(dominant - secondary + 0.5, 0.5, 0.0, 1.0)


def compute_latent_purity(
    *,
    image_entries: list[dict[str, Any]],
    vectors: list[list[float]],
    labels: list[int],
    outlier_indices: list[int],
    emotion_scores: dict[str, float],
    density_values: list[float],
    aesthetic_matches: list[dict[str, Any]],
    color_coherence: float,
) -> dict[str, Any]:
    """Compute pre-spectral purity from latent vectors and detected modalities.

    This step deliberately happens before graph spectral analysis: it measures
    how coherent the moodboard looks from embeddings, clusters, palette and
    model-derived modalities alone.
    """

    cluster_counts = Counter(labels)
    cluster_total = max(1, len(labels))
    cluster_dominance = max(cluster_counts.values(), default=0) / cluster_total
    pair_sims = [
        cosine_similarity(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    avg_similarity = sum(pair_sims) / len(pair_sims) if pair_sims else 1.0
    intra_sims = [
        cosine_similarity(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
        if labels[i] == labels[j]
    ]
    inter_sims = [
        cosine_similarity(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
        if labels[i] != labels[j]
    ]
    intra_similarity = sum(intra_sims) / len(intra_sims) if intra_sims else avg_similarity
    inter_similarity = sum(inter_sims) / len(inter_sims) if inter_sims else 0.0
    cluster_cohesion = clamp_float((intra_similarity + 1.0) / 2.0, 0.5, 0.0, 1.0)
    cluster_separation = clamp_float((intra_similarity - inter_similarity + 1.0) / 2.0, 0.5, 0.0, 1.0)
    outlier_factor = 1.0 - (len(outlier_indices) / max(1, len(image_entries))) * 0.35

    style_terms = observation_counter(image_entries, {"style"})
    affect_terms = observation_counter(image_entries, {"affect"})
    symbolic_terms = observation_counter(image_entries, {"symbol", "object", "texture", "tag", "affect"})
    concept_concentration = _top_share(symbolic_terms, 6, 0.45)
    style_coherence = _top_share(style_terms, 3, 0.45)
    affect_coherence = _top_share(affect_terms, 3, 0.45)
    symbolic_coherence = concept_concentration
    emotional_coherence = clamp_float(max(emotion_scores.values(), default=0.5), 0.5, 0.0, 1.0)
    composition_coherence = clamp_float(
        1.0 - (max(density_values) - min(density_values) if density_values else 0.0),
        0.7,
        0.0,
        1.0,
    )
    semantic_coverage = sum(
        1
        for entry in image_entries
        if entry.get("objects") or entry.get("symbols") or entry.get("affects") or entry.get("styles")
    ) / max(1, len(image_entries))
    modality_coverage = clamp_float(0.42 + semantic_coverage * 0.58, 0.5, 0.0, 1.0)
    aesthetic_margin = _aesthetic_margin(aesthetic_matches)

    latent_cluster_purity = (cluster_cohesion * 0.34) + (cluster_separation * 0.28) + (cluster_dominance * 0.18)
    modality_convergence = (
        (concept_concentration * 0.075)
        + (style_coherence * 0.04)
        + (affect_coherence * 0.035)
        + (emotional_coherence * 0.03)
        + (color_coherence * 0.03)
        + (modality_coverage * 0.02)
    )
    latent_purity = clamp_float(
        (latent_cluster_purity + modality_convergence) * outlier_factor * (0.82 + aesthetic_margin * 0.18),
        0.0,
        0.0,
        1.0,
    )

    return {
        "clusterCounts": cluster_counts,
        "clusterTotal": cluster_total,
        "clusterDominance": cluster_dominance,
        "clusterCohesion": cluster_cohesion,
        "clusterSeparation": cluster_separation,
        "outlierFactor": outlier_factor,
        "aestheticMargin": aesthetic_margin,
        "colorCoherence": clamp_float(color_coherence, 0.0, 0.0, 1.0),
        "styleCoherence": style_coherence,
        "affectCoherence": affect_coherence,
        "symbolicCoherence": symbolic_coherence,
        "emotionalCoherence": emotional_coherence,
        "compositionCoherence": composition_coherence,
        "modalityCoverage": modality_coverage,
        "latentClusterPurity": latent_cluster_purity,
        "modalityConvergence": modality_convergence,
        "latentPurity": latent_purity,
        "purity": latent_purity,
        "hybridation": clamp_float(1.0 - latent_purity, 0.0, 0.0, 1.0),
    }


def finalize_purity(
    *,
    latent_purity: float,
    spectral_analysis: dict[str, Any],
    trained_mood_matches: list[dict[str, Any]],
    aesthetic_margin: float,
    modality_coverage: float,
) -> dict[str, Any]:
    """Fuse latent, spectral and trained-world signals into one purity score."""

    spectral_purity = clamp_float(
        float(spectral_analysis.get("spectralPurityScore", latent_purity)),
        latent_purity,
        0.0,
        1.0,
    )
    harmonicity_score = clamp_float(
        float(spectral_analysis.get("harmonicityScore", spectral_purity)),
        spectral_purity,
        0.0,
        1.0,
    )
    world_mood_confidence = clamp_float(
        float(trained_mood_matches[0].get("confidence", 0.0)) if trained_mood_matches else 0.0,
        0.0,
        0.0,
        1.0,
    )
    final_purity = clamp_float(
        (latent_purity * 0.56)
        + (spectral_purity * 0.24)
        + (harmonicity_score * 0.08)
        + (world_mood_confidence * 0.05)
        + (aesthetic_margin * 0.04)
        + (modality_coverage * 0.03),
        latent_purity,
        0.0,
        1.0,
    )
    updated_spectral = dict(spectral_analysis)
    updated_spectral["inputPurityScore"] = round(latent_purity, 6)
    updated_spectral["finalPurityScore"] = round(final_purity, 6)
    updated_spectral["purityScore"] = round(final_purity, 6)

    return {
        "purity": final_purity,
        "purityScore": final_purity,
        "latentPurity": latent_purity,
        "spectralPurity": spectral_purity,
        "harmonicity": harmonicity_score,
        "harmonyCoherence": harmonicity_score,
        "hybridation": clamp_float(1.0 - final_purity, 0.0, 0.0, 1.0),
        "worldMoodConfidence": world_mood_confidence,
        "spectralAnalysis": updated_spectral,
    }
