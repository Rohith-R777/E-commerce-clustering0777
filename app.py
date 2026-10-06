"""Streamlit dashboard for customer segments and baseline metrics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ecommerce_clustering.explainability import explain_cluster_membership
from ecommerce_clustering.phase1 import build_rfm_features, load_transactions, run_baseline
from ecommerce_clustering.phase6 import build_segment_profiles


def main() -> None:
    st.set_page_config(page_title="Customer Segmentation", layout="wide")
    st.title("E-Commerce Customer Segmentation")
    st.caption("RFM clustering with explainable segment summaries")

    uploaded = st.file_uploader("Upload transactions", type=["csv", "parquet"])
    if uploaded is None:
        st.info("Upload a CSV or Parquet file to begin.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(uploaded.getvalue())
        source = Path(temporary.name)

    transactions = load_transactions(source)
    rfm = build_rfm_features(transactions)
    cluster_count = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=3)
    result = run_baseline(rfm, n_clusters=min(cluster_count, len(rfm) - 1))
    profiles = build_segment_profiles(result.customers)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Customers", len(rfm))
    metric_columns[1].metric("Silhouette", f"{result.metrics['silhouette']:.3f}")
    metric_columns[2].metric("DBI", f"{result.metrics['davies_bouldin']:.3f}")

    st.subheader("Segment profiles")
    st.dataframe(profiles, use_container_width=True, hide_index=True)
    st.bar_chart(profiles.set_index("profile")["customers"])

    st.subheader("Customer assignments")
    st.dataframe(result.customers, use_container_width=True, hide_index=True)

    if st.button("Explain first segment"):
        feature_names = ["recency", "frequency", "monetary"]
        summary, _ = explain_cluster_membership(
            result.model, result.scaler.transform(rfm[feature_names]), feature_names
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
