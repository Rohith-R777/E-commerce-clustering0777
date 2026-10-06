import pandas as pd

from ecommerce_clustering.phase1 import build_clustering_features, build_rfm_features, run_baseline


def test_generic_mixed_schema_is_converted_to_cluster_features() -> None:
    transactions = pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(8)],
            "customer_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "category": ["Home", "Home", "Tech", "Tech", "Food", "Food", "Home", "Food"],
            "price": [10, 12, 100, 110, 25, 30, 15, 35],
            "quantity": [1, 2, 1, 2, 3, 2, 1, 4],
            "order_date": pd.to_datetime(
                ["2024-01-01", "2024-01-03", "2024-02-01", "2024-02-03",
                 "2024-03-01", "2024-03-03", "2024-04-01", "2024-04-03"]
            ),
        }
    )

    features = build_clustering_features(transactions)
    result = run_baseline(features, n_clusters=2)

    assert len(features) == 4
    assert "customer_id" in features.columns
    assert result.customers["cluster"].nunique() == 2


def test_rfm_aggregation_and_baseline_metrics_are_valid() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "invoice_date": pd.to_datetime(
                [
                    "2025-01-01", "2025-01-03", "2025-02-01", "2025-02-04",
                    "2025-03-01", "2025-03-05", "2025-04-01", "2025-04-02",
                    "2025-05-01", "2025-05-02", "2025-06-01", "2025-06-02",
                ]
            ),
            "quantity": [1, 2, 1, 3, 2, 2, 4, 3, 5, 4, 6, 5],
            "unit_price": [10, 12, 20, 18, 30, 28, 40, 42, 50, 52, 60, 58],
        }
    )

    rfm = build_rfm_features(transactions, reference_date="2025-07-01")
    result = run_baseline(rfm, n_clusters=3)

    assert len(rfm) == 6
    assert list(rfm.columns) == ["customer_id", "recency", "frequency", "monetary"]
    assert result.customers["cluster"].nunique() == 3
    assert all(pd.notna(value) for value in result.metrics.values())
    assert result.metrics["silhouette"] > -1
