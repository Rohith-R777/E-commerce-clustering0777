"""Phase 5 adaptive parameter selection using a reproducible reward search."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


def select_configuration(values: np.ndarray, cluster_options: tuple[int, ...] = (2, 3, 4, 5)) -> dict:
    """Select k by the documented quality reward: silhouette minus DBI penalty."""
    values = np.asarray(values, dtype=float)
    candidates = []
    for n_clusters in cluster_options:
        if n_clusters >= len(values):
            continue
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(values)
        silhouette = silhouette_score(values, model.labels_)
        dbi = davies_bouldin_score(values, model.labels_)
        candidates.append(
            {"n_clusters": n_clusters, "silhouette": float(silhouette), "davies_bouldin": float(dbi),
             "reward": float(silhouette - 0.25 * dbi), "labels": model.labels_}
        )
    if not candidates:
        raise ValueError("No valid cluster configuration for the supplied data")
    return max(candidates, key=lambda candidate: candidate["reward"])
