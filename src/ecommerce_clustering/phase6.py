"""Phase 6 segment profiles and inference service helpers."""

from __future__ import annotations

import pandas as pd


def build_segment_profiles(customers: pd.DataFrame) -> pd.DataFrame:
    """Create business-readable summaries for each assigned segment."""
    required = {"cluster", "recency", "frequency", "monetary"}
    missing = required.difference(customers.columns)
    if missing:
        raise ValueError(f"Customer data is missing: {sorted(missing)}")
    profiles = customers.groupby("cluster", as_index=False).agg(
        customers=("cluster", "size"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    profiles["profile"] = profiles.apply(_profile_name, axis=1)
    return profiles


def _profile_name(row: pd.Series) -> str:
    if row["avg_recency"] <= row["avg_recency"] * 0 + 30 and row["avg_monetary"] > 0:
        return "Loyalists"
    if row["avg_recency"] > 90:
        return "At-Risk"
    return "Intermittent"
