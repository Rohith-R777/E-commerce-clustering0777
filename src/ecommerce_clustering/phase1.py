"""Phase 0/1 transaction ingestion, RFM features, and baseline clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {"customer_id", "invoice_date", "quantity", "unit_price"}
ENTITY_COLUMN_NAMES = {
    "customer_id",
    "customerid",
    "customer",
    "user_id",
    "userid",
    "account_id",
    "accountid",
    "client_id",
    "clientid",
}
COLUMN_ALIASES = {
    "CustomerID": "customer_id",
    "InvoiceDate": "invoice_date",
    "Quantity": "quantity",
    "UnitPrice": "unit_price",
}


@dataclass(frozen=True)
class BaselineResult:
    """Customer features, labels, model, and validation metrics."""

    customers: pd.DataFrame
    model: KMeans
    scaler: StandardScaler
    metrics: dict[str, float]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load a tabular dataset; feature builders validate the usable schema."""
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        transactions = pd.read_parquet(source)
    elif source.suffix.lower() in {".csv", ".txt"}:
        try:
            transactions = pd.read_csv(source, encoding="utf-8")
        except UnicodeDecodeError:
            transactions = pd.read_csv(source, encoding="latin-1")
    else:
        raise ValueError("Supported transaction formats are CSV and Parquet")

    transactions = transactions.rename(columns=COLUMN_ALIASES)
    if transactions.empty or len(transactions.columns) < 2:
        raise ValueError("The input dataset must contain rows and at least two columns")
    return transactions


def build_clustering_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build legacy RFM features or infer features from a mixed-schema dataset."""
    if REQUIRED_COLUMNS.issubset(transactions.columns):
        return build_rfm_features(transactions)
    return build_generic_features(transactions)


def build_generic_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Infer entity-level numeric features from numeric, date, and categorical data."""
    if transactions.empty:
        raise ValueError("The input dataset must not be empty")

    frame = transactions.copy()
    entity_column = _find_entity_column(frame)
    if entity_column is None:
        entity_column = "_row_id"
        frame[entity_column] = np.arange(len(frame))
    frame = frame.dropna(subset=[entity_column]).copy()
    if frame.empty:
        raise ValueError("No rows contain a usable entity identifier")

    entity_values = frame[entity_column].astype(str)
    grouped = pd.DataFrame(index=pd.Index(entity_values.unique(), name="_entity"))
    feature_columns: list[str] = []

    for column in frame.columns:
        if column == entity_column or _looks_like_identifier(column, frame[column]):
            continue

        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.9 and numeric.nunique(dropna=True) > 1:
            name = f"{column}__mean"
            grouped[name] = numeric.groupby(entity_values).mean()
            feature_columns.append(name)
            continue

        if _looks_like_date(column, series):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        else:
            parsed = pd.Series(pd.NaT, index=series.index)
        if parsed.notna().mean() >= 0.7:
            end_date = parsed.max()
            recency = (end_date - parsed).dt.total_seconds() / 86400
            name = f"{column}__recency_days"
            grouped[name] = recency.groupby(entity_values).min()
            feature_columns.append(name)
            continue

        values = series.astype("string").fillna("__missing__")
        if values.nunique() <= 20:
            encoded = pd.get_dummies(values, prefix=column, dtype=float)
            encoded.index = entity_values
            proportions = encoded.groupby(level=0).mean()
            for name in proportions.columns:
                grouped[name] = proportions[name]
                feature_columns.append(name)

    if not feature_columns:
        raise ValueError("No usable numeric, date, or low-cardinality categorical features found")

    result = grouped.reset_index().rename(columns={"_entity": entity_column})
    result[feature_columns] = result[feature_columns].replace([np.inf, -np.inf], np.nan)
    result[feature_columns] = result[feature_columns].fillna(result[feature_columns].median())
    result = result.dropna(subset=feature_columns)
    if len(result) < 4:
        raise ValueError("At least four usable entities are required for clustering")
    return result[[entity_column, *feature_columns]].reset_index(drop=True)


def _find_entity_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized in ENTITY_COLUMN_NAMES:
            return column
    return None


def _looks_like_identifier(column: str, series: pd.Series) -> bool:
    normalized = str(column).strip().lower().replace(" ", "_")
    if any(token in normalized for token in ("id", "code", "uuid")):
        return True
    return series.nunique(dropna=True) >= max(20, int(len(series) * 0.95)) and not pd.api.types.is_numeric_dtype(series)


def _looks_like_date(column: str, series: pd.Series) -> bool:
    normalized = str(column).strip().lower()
    return pd.api.types.is_datetime64_any_dtype(series) or any(
        token in normalized for token in ("date", "time", "timestamp")
    )


def build_rfm_features(
    transactions: pd.DataFrame,
    reference_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate transaction rows into recency, frequency, and monetary features."""
    missing = REQUIRED_COLUMNS.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")

    frame = transactions.copy()
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="raise")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="raise")
    frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="raise")
    frame = frame.dropna(subset=["customer_id", "invoice_date"])
    frame["amount"] = frame["quantity"] * frame["unit_price"]

    end_date = pd.Timestamp(reference_date) if reference_date is not None else frame["invoice_date"].max()
    rfm = (
        frame.groupby("customer_id", as_index=False)
        .agg(
            last_purchase=("invoice_date", "max"),
            frequency=("invoice_date", "count"),
            monetary=("amount", "sum"),
        )
    )
    rfm["recency"] = (end_date - rfm["last_purchase"]).dt.days.clip(lower=0)
    return rfm[["customer_id", "recency", "frequency", "monetary"]].sort_values(
        "customer_id"
    ).reset_index(drop=True)


def run_baseline(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> BaselineResult:
    """Fit standardized K-Means++ and calculate baseline clustering metrics."""
    feature_names = ["recency", "frequency", "monetary"]
    if not set(feature_names).issubset(rfm.columns):
        feature_names = list(rfm.columns[1:])
    if not feature_names:
        raise ValueError("Clustering data must contain numeric features")
    if len(rfm) <= n_clusters:
        raise ValueError("The number of customers must exceed n_clusters")

    features = rfm[feature_names].astype(float).replace([np.inf, -np.inf], np.nan)
    if features.isna().any().any():
        raise ValueError("RFM features must be finite")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=random_state)
    labels = model.fit_predict(scaled)
    customers = rfm.copy()
    customers["cluster"] = labels
    metrics = {
        "silhouette": float(silhouette_score(scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
        "inertia": float(model.inertia_),
    }
    return BaselineResult(customers=customers, model=model, scaler=scaler, metrics=metrics)
