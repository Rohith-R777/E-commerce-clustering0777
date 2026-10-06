"""Phase 3 clustering engine with scalable local alternatives."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def cluster_embeddings(values: np.ndarray, method: str = "kmeans", n_clusters: int = 3) -> dict:
    """Cluster embeddings and return labels plus comparable validation metrics."""
    values = np.asarray(values, dtype=float)
    if method == "kmeans":
        estimator = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    elif method == "birch":
        estimator = Birch(n_clusters=n_clusters)
    else:
        raise ValueError("method must be 'kmeans' or 'birch'")
    labels = estimator.fit_predict(values)
    return {
        "method": method,
        "labels": labels,
        "model": estimator,
        "silhouette": float(silhouette_score(values, labels)),
        "davies_bouldin": float(davies_bouldin_score(values, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(values, labels)),
    }
