"""Latent vector clustering and projection utilities."""

from __future__ import annotations

import math

from src.moodboard.analysis.scoring import cosine_similarity
from src.moodboard.core.cache import ANALYSIS_RUNTIME_LOCK

def cluster(vectors: list[list[float]]) -> tuple[list[int], list[int]]:
    return cluster_vectors(vectors)


def vector_projection_2d(vectors: list[list[float]]) -> tuple[list[tuple[float, float]], str]:
    """Project embedding vectors into 2D while preserving latent distances.

    UMAP is preferred for medium-sized graphs, with TSNE/MDS/PCA fallbacks so
    the graph remains usable when optional dependencies fail.
    """

    clean_vectors = [vector for vector in vectors if vector]
    if len(clean_vectors) != len(vectors) or not vectors:
        return [(0.0, 0.0) for _ in vectors], "none"
    if len(vectors) == 1:
        return [(0.0, 0.0)], "single"
    try:
        import numpy as np  # type: ignore

        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-9)
        method = "pca"
        coords = None
        if len(vectors) >= 6:
            try:
                with ANALYSIS_RUNTIME_LOCK:
                    import umap  # type: ignore

                    reducer = umap.UMAP(
                        n_components=2,
                        metric="cosine",
                        n_neighbors=max(2, min(12, len(vectors) - 1)),
                        min_dist=0.34,
                        spread=1.85,
                        random_state=42,
                    )
                    coords = reducer.fit_transform(matrix)
                method = "umap-cosine"
            except Exception:
                coords = None
        if coords is None and len(vectors) >= 4:
            try:
                from sklearn.manifold import TSNE  # type: ignore

                perplexity = max(2, min(12, (len(vectors) - 1) // 3 or 2))
                reducer = TSNE(
                    n_components=2,
                    metric="cosine",
                    perplexity=perplexity,
                    init="pca",
                    learning_rate="auto",
                    random_state=42,
                )
                coords = reducer.fit_transform(matrix)
                method = "tsne-cosine"
            except Exception:
                coords = None
        if coords is None and len(vectors) >= 3:
            try:
                from sklearn.manifold import MDS  # type: ignore

                similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)
                distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarities))
                reducer = MDS(n_components=2, dissimilarity="precomputed", normalized_stress="auto", random_state=42)
                coords = reducer.fit_transform(distances)
                method = "mds-cosine"
            except Exception:
                coords = None
        if coords is None:
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
            if vt.shape[0] >= 2:
                coords = centered @ vt[:2].T
            else:
                coords = np.column_stack([centered[:, 0], np.zeros(len(vectors))])
            method = "pca"

        coords = np.asarray(coords, dtype=float)
        coords = coords - coords.mean(axis=0, keepdims=True)
        max_radius = float(np.sqrt((coords**2).sum(axis=1)).max())
        if max_radius <= 1e-9:
            angles = np.linspace(0, 2 * math.pi, len(vectors), endpoint=False)
            coords = np.column_stack([np.cos(angles), np.sin(angles)])
            max_radius = 1.0
            method = "fallback-circle"
        coords = coords / max_radius
        radii = np.sqrt((coords**2).sum(axis=1))
        if len(vectors) >= 12:
            expanded_radii = np.power(np.maximum(radii, 1e-9), 0.86) * 1.08
            coords = coords * (expanded_radii / np.maximum(radii, 1e-9))[:, None]
        else:
            coords = coords * 1.02
        return [(round(float(x), 4), round(float(y), 4)) for x, y in coords], method
    except Exception:
        coords = []
        for idx in range(len(vectors)):
            angle = 2.0 * math.pi * idx / max(1, len(vectors))
            coords.append((round(math.cos(angle) * 0.78, 4), round(math.sin(angle) * 0.78, 4)))
        return coords, "fallback-circle"


def cluster_vectors(vectors: list[list[float]]) -> tuple[list[int], list[int]]:
    """Cluster normalized embeddings and flag visually/aesthetically isolated items."""

    if not vectors:
        return [], []
    if len(vectors) == 1:
        return [0], []
    hdbscan_outliers: list[int] = []
    try:
        import hdbscan  # type: ignore
        import numpy as np  # type: ignore

        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-9)
        distance = np.clip(1.0 - np.matmul(matrix, matrix.T), 0.0, 2.0)
        min_cluster_size = max(2, min(5, int(round(math.sqrt(len(vectors))))))
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
            min_samples=1,
            cluster_selection_method="eom",
            allow_single_cluster=False,
        )
        raw_labels = [int(value) for value in clusterer.fit_predict(distance)]
        non_noise = sorted({label for label in raw_labels if label >= 0})
        if len(non_noise) >= 2:
            remap = {label: idx for idx, label in enumerate(non_noise)}
            fallback_cluster = len(remap)
            labels = []
            for idx, label in enumerate(raw_labels):
                if label < 0:
                    hdbscan_outliers.append(idx)
                    labels.append(fallback_cluster + (idx % 2))
                else:
                    labels.append(remap[label])
        else:
            raise ValueError("HDBSCAN found fewer than two stable clusters")
    except Exception:
        labels = []

    if not labels:
        try:
            from sklearn.cluster import KMeans  # type: ignore
            from sklearn.metrics import silhouette_score  # type: ignore

            import numpy as np  # type: ignore

            matrix = np.asarray(vectors, dtype=float)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-9)
            min_candidate = 3 if len(vectors) >= 8 else 2
            max_clusters = max(1, min(6, len(vectors) - 1, int(round(math.sqrt(len(vectors)) * 1.7))))
            cluster_count = 1
            best_adjusted_score = -1.0
            best_labels: list[int] | None = None
            target_clusters = max(min_candidate, min(max_clusters, int(round(math.sqrt(len(vectors)) * 1.35))))
            for candidate in range(min_candidate, max_clusters + 1):
                model = KMeans(n_clusters=candidate, n_init="auto", random_state=42)
                candidate_labels = [int(value) for value in model.fit_predict(matrix)]
                if len(set(candidate_labels)) < 2:
                    continue
                score = float(silhouette_score(matrix, candidate_labels, metric="cosine"))
                adjusted_score = score - abs(candidate - target_clusters) * 0.018
                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_labels = candidate_labels
                    cluster_count = candidate
            if cluster_count <= 1:
                labels = [0] * len(vectors)
            elif best_labels is not None:
                labels = best_labels
            else:
                model = KMeans(n_clusters=cluster_count, n_init="auto", random_state=42)
                labels = [int(value) for value in model.fit_predict(matrix)]
        except Exception:
            labels = [idx % max(1, min(3, len(vectors))) for idx in range(len(vectors))]

    outliers: list[int] = list(hdbscan_outliers)
    for idx, vector in enumerate(vectors):
        similarities = [
            cosine_similarity(vector, other)
            for other_idx, other in enumerate(vectors)
            if other_idx != idx
        ]
        if idx not in outliers and similarities and sum(similarities) / len(similarities) < 0.58:
            outliers.append(idx)
    return labels, outliers
