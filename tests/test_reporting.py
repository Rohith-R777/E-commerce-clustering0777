import numpy as np
import pandas as pd

from ecommerce_clustering.explainability import explain_cluster_membership
from ecommerce_clustering.phase1 import build_rfm_features, run_baseline
from ecommerce_clustering.report import generate_report


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "invoice_date": pd.date_range("2025-01-01", periods=12, freq="10D"),
            "quantity": [1, 2, 1, 3, 2, 2, 4, 3, 5, 4, 6, 5],
            "unit_price": [10, 12, 20, 18, 30, 28, 40, 42, 50, 52, 60, 58],
        }
    )


def test_shap_summary_and_report_artifacts(tmp_path) -> None:
    transactions = _transactions()
    rfm = build_rfm_features(transactions)
    result = run_baseline(rfm)
    values = result.scaler.transform(rfm[["recency", "frequency", "monetary"]])
    summary, shap_values = explain_cluster_membership(
        result.model, values, ["recency", "frequency", "monetary"], max_samples=6, background_size=6
    )
    input_path = tmp_path / "transactions.csv"
    transactions.to_csv(input_path, index=False)
    metrics = generate_report(input_path, tmp_path / "results")

    assert summary.iloc[0]["feature"] in {"recency", "frequency", "monetary"}
    assert shap_values.shape == (6, 3)
    assert metrics["customer_count"] == 6
    assert (tmp_path / "results" / "report.md").exists()
