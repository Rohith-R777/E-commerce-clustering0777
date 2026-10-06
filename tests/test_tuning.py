import numpy as np

from ecommerce_clustering.hybrid import select_business_configuration, tune_dbscan


def test_dbscan_tuning_returns_retention_candidates() -> None:
    values = np.vstack(
        [
            np.random.default_rng(42).normal(0, 0.2, (10, 3)),
            np.random.default_rng(43).normal(5, 0.2, (10, 3)),
            np.random.default_rng(44).normal(10, 0.2, (10, 3)),
        ]
    )
    candidates = tune_dbscan(values, min_samples_range=(2, 3))
    selected, result = select_business_configuration(values)

    assert candidates
    assert selected["retention_pct"] >= 80
    assert result.metrics["outliers"] <= 6