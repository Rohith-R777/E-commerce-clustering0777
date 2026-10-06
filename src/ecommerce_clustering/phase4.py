"""Phase 4 drift monitoring and incremental clustering updates."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate PSI for two one-dimensional distributions."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


class IncrementalClusterer:
    """MiniBatchKMeans wrapper for online customer embedding updates."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state, batch_size=1024, n_init=10
        )

    def fit(self, values: np.ndarray) -> "IncrementalClusterer":
        self.model.partial_fit(values)
        return self

    def update(self, values: np.ndarray) -> np.ndarray:
        self.model.partial_fit(values)
        return self.model.predict(values)
