"""Spectral aesthetic analysis over the moodboard graph."""

from __future__ import annotations

import math
from typing import Any


def clamp_float(value: float, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(low, min(high, number))


def analyze(graph: dict[str, Any], dominant_name: str, purity_score: float) -> dict[str, Any]:
    return spectral_aesthetic_analysis(graph, dominant_name, purity_score)


def spectral_aesthetic_analysis(graph: dict[str, Any], dominant_name: str, purity_score: float) -> dict[str, Any]:
    """Compute Laplacian eigenvalue metrics for latent aesthetic harmony."""

    raw_nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    nodes = [
        node
        for node in raw_nodes
        if str(node.get("type", "")) != "color" or float(node.get("weight", 0.0) or 0.0) >= 0.055
    ]
    if len(nodes) < 2:
        nodes = raw_nodes
    node_count = len(nodes)
    if node_count < 2:
        return {
            "eigenvalues": [],
            "energy": [],
            "spectralGap": 0.0,
            "normalizedSpectralGap": 0.0,
            "harmonicityScore": 0.0,
            "eigenEntropyHarmonicity": 0.0,
            "lowFrequencyConcentration": 0.0,
            "dissonanceScore": 0.0,
            "hybridizationScore": 0.0,
            "spectralPurityScore": 0.0,
            "spectralFlatness": 0.0,
            "structuralCoherence": 0.0,
            "purityScore": round(purity_score, 6),
            "dominantAestheticFrequency": 0.0,
            "aestheticRegimeCount": 0,
            "componentCount": 0,
            "distanceToTargetHarmony": 1.0,
            "regimePeaks": [],
            "dominantAesthetic": dominant_name,
            "spectralNodeCount": node_count,
            "rawNodeCount": len(raw_nodes),
            "interpretation": "Waiting for enough graph structure.",
            "status": "insufficient-graph",
        }

    try:
        import numpy as np  # type: ignore

        index_by_id = {str(node.get("id")): idx for idx, node in enumerate(nodes)}
        adjacency = np.zeros((node_count, node_count), dtype=float)
        edge_type_weight = {
            "image_similarity": 1.0,
            "attention_similarity": 0.74,
            "aesthetic_match": 0.92,
            "co_occurrence": 0.78,
            "style_affinity": 0.70,
            "emotion_affinity": 0.66,
            "texture_affinity": 0.58,
            "composition_affinity": 0.56,
            "color_affinity": 0.38,
        }
        for edge in edges:
            source = index_by_id.get(str(edge.get("source")))
            target = index_by_id.get(str(edge.get("target")))
            if source is None or target is None or source == target:
                continue
            edge_type = str(edge.get("type", ""))
            multiplier = edge_type_weight.get(edge_type, 0.64)
            weight = clamp_float(float(edge.get("weight", 0.0)) * multiplier, 0.0, 0.0, 1.0)
            adjacency[source, target] = max(adjacency[source, target], weight)
            adjacency[target, source] = max(adjacency[target, source], weight)

        degree_values = adjacency.sum(axis=1)
        positive_degree = degree_values > 1e-9
        laplacian = np.diag(degree_values) - adjacency
        raw_eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.maximum(raw_eigenvalues, 0.0)
        max_eigenvalue = float(eigenvalues[-1]) if node_count else 0.0
        normalized_eigenvalues = eigenvalues / max(max_eigenvalue, 1e-9)
        total_energy = float(normalized_eigenvalues.sum())
        if total_energy <= 1e-9:
            energy = np.zeros(node_count, dtype=float)
            entropy = 0.0
        else:
            energy = normalized_eigenvalues / total_energy
            entropy = float(-np.sum(energy * np.log(energy + 1e-9)))

        max_entropy = math.log(max(2, node_count))
        spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if node_count > 1 else 0.0
        normalized_gap = spectral_gap / max(max_eigenvalue, 1e-9)
        high_start = max(1, int(math.ceil(node_count * 0.62)))
        zero_threshold = max(1e-7, max_eigenvalue * 1e-6)
        positive_eigenvalues = normalized_eigenvalues[normalized_eigenvalues > max(1e-7, zero_threshold / max(max_eigenvalue, 1e-9))]
        tau = max(0.08, float(np.median(positive_eigenvalues)) * 0.55) if len(positive_eigenvalues) else 0.28
        mode_strength = np.exp(-normalized_eigenvalues / tau)
        mode_energy = mode_strength / max(float(mode_strength.sum()), 1e-9)
        mode_entropy = float(-np.sum(mode_energy * np.log(mode_energy + 1e-9)))
        eigenvalue_energy = energy
        energy = mode_energy
        entropy = mode_entropy
        dispersion = clamp_float(entropy / max_entropy, 0.0, 0.0, 1.0)
        eigen_entropy_harmonicity = clamp_float(1.0 - dispersion, 0.0, 0.0, 1.0)
        dissonance = float(energy[high_start:].sum()) if node_count else 0.0
        low_band_end = max(2, min(node_count, int(math.ceil(math.sqrt(node_count))) + 1))
        low_band_concentration = float(mode_energy[:low_band_end].sum())

        visited: set[int] = set()
        component_count = 0
        for start in range(node_count):
            if start in visited:
                continue
            component_count += 1
            stack = [start]
            visited.add(start)
            while stack:
                current = stack.pop()
                neighbors = np.nonzero(adjacency[current] > 1e-9)[0]
                for neighbor in neighbors:
                    neighbor_idx = int(neighbor)
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        stack.append(neighbor_idx)

        positive_energy = energy[energy > 1e-9]
        if len(positive_energy):
            spectral_flatness = float(np.exp(np.mean(np.log(positive_energy + 1e-9))) / max(float(np.mean(positive_energy)), 1e-9))
        else:
            spectral_flatness = 0.0
        component_penalty = clamp_float((component_count - 1) / max(1, min(8, node_count - 1)), 0.0, 0.0, 1.0)
        structural_coherence = clamp_float(
            (low_band_concentration * 0.34)
            + (eigen_entropy_harmonicity * 0.24)
            + (clamp_float(normalized_gap, 0.0, 0.0, 1.0) * 0.18)
            + ((1.0 - clamp_float(dissonance, 0.0, 0.0, 1.0)) * 0.16)
            + ((1.0 - clamp_float(spectral_flatness, 0.0, 0.0, 1.0)) * 0.08),
            0.0,
            0.0,
            1.0,
        )
        harmonicity = clamp_float(structural_coherence * (1.0 - component_penalty * 0.28), 0.0, 0.0, 1.0)
        spectral_purity_score = clamp_float(
            (harmonicity * 0.48)
            + (low_band_concentration * 0.18)
            + (eigen_entropy_harmonicity * 0.16)
            + ((1.0 - clamp_float(dissonance, 0.0, 0.0, 1.0)) * 0.12)
            + (clamp_float(normalized_gap, 0.0, 0.0, 1.0) * 0.06),
            0.0,
            0.0,
            1.0,
        )
        dominant_index = 0
        if node_count > 1 and total_energy > 1e-9:
            dominant_index = int(np.argmax(energy[1:]) + 1)
        dominant_frequency = dominant_index / max(1, node_count - 1)

        peak_candidates: list[dict[str, Any]] = []
        max_energy = float(energy.max()) if node_count else 0.0
        significant_floor = max(max_energy * 0.18, 1.0 / max(3, node_count) * 0.55)
        for idx in range(1, node_count):
            previous_energy = float(energy[idx - 1]) if idx > 0 else 0.0
            current_energy = float(energy[idx])
            next_energy = float(energy[idx + 1]) if idx + 1 < node_count else 0.0
            if current_energy >= significant_floor and current_energy >= previous_energy and current_energy >= next_energy:
                peak_candidates.append(
                    {
                        "index": idx,
                        "frequency": round(idx / max(1, node_count - 1), 4),
                        "eigenvalue": round(float(eigenvalues[idx]), 6),
                        "energy": round(current_energy, 6),
                    }
                )
        if not peak_candidates and dominant_index:
            peak_candidates.append(
                {
                    "index": dominant_index,
                    "frequency": round(dominant_frequency, 4),
                    "eigenvalue": round(float(eigenvalues[dominant_index]), 6),
                    "energy": round(float(energy[dominant_index]), 6),
                }
            )
        peak_candidates.sort(key=lambda item: item["energy"], reverse=True)
        strong_peak_count = sum(
            1
            for item in peak_candidates
            if float(item.get("energy", 0.0)) >= max_energy * 0.30 and int(item.get("index", 0)) <= high_start
        )
        regime_count = max(component_count, min(6, strong_peak_count or 1))
        hybridization_score = clamp_float((regime_count - 1) / 7.0 + (1.0 - normalized_gap) * 0.16 + spectral_flatness * 0.12, 0.0, 0.0, 1.0)
        distance_to_target = clamp_float(abs(0.88 - spectral_purity_score) + dissonance * 0.20 + max(0, regime_count - 1) * 0.038, 0.0, 0.0, 1.0)

        if spectral_purity_score >= 0.74 and harmonicity >= 0.66:
            interpretation = f"Concentrated spectrum: dominant {dominant_name} regime with strong latent harmony."
        elif regime_count in (2, 3):
            interpretation = f"Stable hybrid spectrum: {regime_count} aesthetic regimes are co-present."
        elif spectral_flatness >= 0.78 or dissonance >= 0.48:
            interpretation = "Noisy spectrum: high-frequency energy suggests aesthetic dissonance or outliers."
        else:
            interpretation = "Fragmented spectrum: several frequencies compete without a single dominant regime."

        return {
            "eigenvalues": [round(float(value), 6) for value in eigenvalues.tolist()],
            "normalizedEigenvalues": [round(float(value), 6) for value in normalized_eigenvalues.tolist()],
            "energy": [round(float(value), 6) for value in energy.tolist()],
            "eigenvalueEnergy": [round(float(value), 6) for value in eigenvalue_energy.tolist()],
            "spectralGap": round(spectral_gap, 6),
            "normalizedSpectralGap": round(clamp_float(normalized_gap, 0.0, 0.0, 1.0), 6),
            "harmonicityScore": round(harmonicity, 6),
            "eigenEntropyHarmonicity": round(eigen_entropy_harmonicity, 6),
            "lowFrequencyConcentration": round(clamp_float(low_band_concentration, 0.0, 0.0, 1.0), 6),
            "dissonanceScore": round(clamp_float(dissonance, 0.0, 0.0, 1.0), 6),
            "hybridizationScore": round(hybridization_score, 6),
            "spectralPurityScore": round(spectral_purity_score, 6),
            "spectralFlatness": round(clamp_float(spectral_flatness, 0.0, 0.0, 1.0), 6),
            "structuralCoherence": round(structural_coherence, 6),
            "inputPurityScore": round(purity_score, 6),
            "purityScore": round(purity_score, 6),
            "dominantAestheticFrequency": round(dominant_frequency, 6),
            "aestheticRegimeCount": int(regime_count),
            "componentCount": int(component_count),
            "distanceToTargetHarmony": round(distance_to_target, 6),
            "regimePeaks": peak_candidates[:6],
            "dominantAesthetic": dominant_name,
            "spectralNodeCount": node_count,
            "rawNodeCount": len(raw_nodes),
            "interpretation": interpretation,
            "status": "unnormalized-laplacian-eigh-v2",
        }
    except Exception as exc:
        return {
            "eigenvalues": [],
            "energy": [],
            "spectralGap": 0.0,
            "normalizedSpectralGap": 0.0,
            "harmonicityScore": 0.0,
            "eigenEntropyHarmonicity": 0.0,
            "lowFrequencyConcentration": 0.0,
            "dissonanceScore": 0.0,
            "hybridizationScore": 0.0,
            "spectralPurityScore": 0.0,
            "spectralFlatness": 0.0,
            "structuralCoherence": 0.0,
            "purityScore": round(purity_score, 6),
            "dominantAestheticFrequency": 0.0,
            "aestheticRegimeCount": 0,
            "componentCount": 0,
            "distanceToTargetHarmony": 1.0,
            "regimePeaks": [],
            "dominantAesthetic": dominant_name,
            "spectralNodeCount": 0,
            "rawNodeCount": len(graph.get("nodes", []) or []),
            "interpretation": f"Spectral analysis unavailable: {exc}",
            "status": "spectral-error",
        }
