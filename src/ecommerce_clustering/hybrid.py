"""Hybrid DBSCAN noise filtering followed by K-Means clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class HybridResult:
    """Metrics and labels for the retained customers."""

    labels: np.ndarray
    core_mask: np.ndarray
    dbscan_labels: np.ndarray
    metrics: dict[str, float]
    eps: float


def tune_dbscan(
    values: np.ndarray,
    min_samples_range: tuple[int, ...] = (2, 3, 4, 5),
    percentiles: tuple[int, ...] = (50, 75, 90, 95),
) -> list[dict[str, float | int]]:
    """Evaluate DBSCAN settings using k-distance-derived epsilon candidates."""
    values = np.asarray(values, dtype=float)
    if len(values) < max(min_samples_range):
        raise ValueError("values must contain at least max(min_samples_range) rows")

    results: list[dict[str, float | int]] = []
    for min_samples in min_samples_range:
        distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(values).kneighbors(values)
        candidates = np.unique(np.round(np.percentile(distances[:, -1], percentiles), 4))
        for eps in candidates:
            labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(values)
            noise_count = int((labels == -1).sum())
            cluster_count = len(set(labels)) - int(-1 in labels)
            results.append(
                {
                    "eps": float(eps),
                    "min_samples": min_samples,
                    "noise_pct": noise_count / len(values) * 100,
                    "clusters": cluster_count,
                    "retention_pct": (len(values) - noise_count) / len(values) * 100,
                }
            )
    return results


def select_business_configuration(
    values: np.ndarray,
    n_clusters: int = 3,
    minimum_retention_pct: float = 80,
) -> tuple[dict[str, float | int], HybridResult]:
    """Select the best valid hybrid setting while enforcing customer coverage."""
    candidates = tune_dbscan(values)
    scored: list[tuple[dict[str, float | int], HybridResult]] = []
    for candidate in candidates:
        if candidate["retention_pct"] < minimum_retention_pct or candidate["clusters"] < 2:
            continue
        try:
            result = run_dbscan_kmeans(
                values,
                n_clusters=n_clusters,
                eps=float(candidate["eps"]),
                min_samples=int(candidate["min_samples"]),
            )
        except ValueError:
            continue
        candidate = {
            **candidate,
            "silhouette": result.metrics["silhouette"],
            "davies_bouldin": result.metrics["davies_bouldin"],
        }
        scored.append((candidate, result))
    if not scored:
        raise ValueError("No DBSCAN configuration met the minimum retention requirement")
    return max(scored, key=lambda pair: float(pair[0]["silhouette"]))


def run_dbscan_kmeans(
    values: np.ndarray,
    n_clusters: int = 3,
    eps: float = 0.35,
    min_samples: int = 2,
    random_state: int = 42,
) -> HybridResult:
    """Remove DBSCAN noise, then score K-Means on the retained customers.

    Metrics intentionally describe only retained customers. Noise remains
    available through ``core_mask`` and ``dbscan_labels`` for reporting.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) <= n_clusters:
        raise ValueError("values must contain more rows than n_clusters")

    dbscan_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(values)
    core_mask = dbscan_labels != -1
    core_values = values[core_mask]
    if len(core_values) <= n_clusters:
        raise ValueError("DBSCAN retained too few customers for K-Means")

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(core_values)
    if len(np.unique(labels)) < 2:
        raise ValueError("K-Means produced fewer than two clusters")
    metrics = {
        "silhouette": float(silhouette_score(core_values, labels)),
        "davies_bouldin": float(davies_bouldin_score(core_values, labels)),
        "retained_customers": float(len(core_values)),
        "outliers": float((~core_mask).sum()),
    }
    return HybridResult(labels, core_mask, dbscan_labels, metrics, eps)
