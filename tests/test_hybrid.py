import numpy as np

from ecommerce_clustering.hybrid import run_dbscan_kmeans


def test_dbscan_kmeans_returns_comparable_metrics() -> None:
    values = np.vstack(
        [
            np.random.default_rng(42).normal(0, 0.1, (8, 3)),
            np.random.default_rng(43).normal(5, 0.1, (8, 3)),
            np.random.default_rng(44).normal(10, 0.1, (8, 3)),
        ]
    )
    result = run_dbscan_kmeans(values, eps=0.5, min_samples=2)

    assert result.core_mask.shape == (24,)
    assert result.metrics["outliers"] >= 0
    assert result.metrics["silhouette"] > 0.8
    assert result.metrics["davies_bouldin"] < 1
