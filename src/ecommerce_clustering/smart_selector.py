"""Data-driven selection of a customer-clustering pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import skew
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def profile_dataset(values: np.ndarray) -> dict[str, Any]:
    """Measure sample size, dimensionality, outliers, skew, density, and tendency."""
    values = _validate_values(values)
    n_samples, n_features = values.shape
    q1, q3 = np.percentile(values, [25, 75], axis=0)
    iqr = q3 - q1
    outlier_mask = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).any(axis=1)
    feature_skew = np.nan_to_num(np.abs(skew(values, axis=0, bias=False)), nan=0.0)
    neighbor_count = min(5, n_samples - 1)
    distances = NearestNeighbors(n_neighbors=neighbor_count).fit(values).kneighbors(values)[0]
    knn_distances = distances[:, -1]
    density_cv = float(knn_distances.std() / (knn_distances.mean() + 1e-9))
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "outlier_ratio": round(float(outlier_mask.mean()), 4),
        "mean_skew": round(float(feature_skew.mean()), 4),
        "density_cv": round(density_cv, 4),
        "hopkins": round(_hopkins_statistic(values), 4),
    }


def _hopkins_statistic(values: np.ndarray, sample_size: int = 30) -> float:
    """Estimate cluster tendency; values near 0.5 are weakly structured."""
    n_samples = len(values)
    sample_count = min(sample_size, n_samples - 1)
    rng = np.random.default_rng(42)
    nearest = NearestNeighbors(n_neighbors=2).fit(values)
    indices = rng.choice(n_samples, sample_count, replace=False)
    real_distances = nearest.kneighbors(values[indices])[0][:, 1]
    random_points = rng.uniform(values.min(axis=0), values.max(axis=0), (sample_count, values.shape[1]))
    random_distances = nearest.kneighbors(random_points, n_neighbors=1)[0][:, 0]
    return float(random_distances.sum() / (real_distances.sum() + random_distances.sum() + 1e-9))


def recommend_algorithm(profile: dict[str, Any]) -> dict[str, Any]:
    """Choose a pipeline and explain the decision using explicit rules."""
    n = profile["n_samples"]
    features = profile["n_features"]
    outlier_ratio = profile["outlier_ratio"]
    density_cv = profile["density_cv"]
    hopkins = profile["hopkins"]

    if hopkins < 0.5:
        return {"algorithm": "none", "config": {}, "reason": f"Hopkins={hopkins:.2f} < 0.5; weak cluster tendency"}
    if features > 10:
        return {
            "algorithm": "autoencoder_kmeans",
            "config": {"latent_dim": 8, "n_clusters": 3},
            "reason": f"features={features} > 10; reduce dimensionality before K-Means",
        }
    if n < 100:
        if outlier_ratio > 0.10:
            return {
                "algorithm": "kmeans_flag_outliers",
                "config": {"n_clusters": 3},
                "reason": f"small dataset (n={n}) with {outlier_ratio:.1%} IQR outliers; preserve all customers",
            }
        return {
            "algorithm": "kmeans",
            "config": {"n_clusters": 3},
            "reason": f"small dataset (n={n}) with low outlier ratio; DBSCAN density is unreliable",
        }
    if n < 1000 and outlier_ratio >= 0.15:
        return {
            "algorithm": "hybrid_dbscan_kmeans",
            "config": {"eps_pct": 75, "min_samples": 3, "n_clusters": 3},
            "reason": f"medium dataset with {outlier_ratio:.1%} outliers; density filtering is viable",
        }
    if n >= 1000 and density_cv > 0.5:
        return {
            "algorithm": "hybrid_dbscan_kmeans",
            "config": {"eps_pct": 90, "min_samples": 5, "n_clusters": 3},
            "reason": f"large dataset with variable density (CV={density_cv:.2f})",
        }
    return {
        "algorithm": "kmeans",
        "config": {"n_clusters": 3},
        "reason": f"fallback for n={n} and outlier ratio={outlier_ratio:.1%}",
    }


def run_selected_pipeline(values: np.ndarray, decision: dict[str, Any]) -> dict[str, Any]:
    """Execute the selected pipeline and return labels, metrics, and retention."""
    values = _validate_values(values)
    algorithm = decision["algorithm"]
    cluster_count = decision.get("config", {}).get("n_clusters", 3)
    if algorithm == "none":
        return {"labels": None, "metrics": {}, "retention": 0.0, "outlier_flags": None}

    model_values = values
    if algorithm == "autoencoder_kmeans":
        latent_dim = decision.get("config", {}).get("latent_dim", 8)
        components = min(latent_dim, values.shape[1], len(values) - 1)
        model_values = PCA(n_components=components, random_state=42).fit_transform(values)
    model = KMeans(n_clusters=cluster_count, n_init=20, random_state=42)
    labels = model.fit_predict(model_values)
    outlier_flags = np.zeros(len(values), dtype=bool)
    if algorithm == "kmeans_flag_outliers":
        q1, q3 = np.percentile(values, [25, 75], axis=0)
        iqr = q3 - q1
        outlier_flags = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).any(axis=1)
    elif algorithm == "hybrid_dbscan_kmeans":
        config = decision["config"]
        distances = NearestNeighbors(n_neighbors=config["min_samples"]).fit(values).kneighbors(values)[0]
        eps = float(np.percentile(distances[:, -1], config["eps_pct"]))
        outlier_flags = DBSCAN(eps=eps, min_samples=config["min_samples"]).fit_predict(values) == -1
    return {
        "labels": labels,
        "metrics": _score(model_values, labels),
        "retention": 1.0,
        "outlier_flags": outlier_flags,
    }


def smart_cluster(values: np.ndarray) -> dict[str, Any]:
    """Profile, recommend, and execute a clustering pipeline in one call."""
    values = _validate_values(values)
    profile = profile_dataset(values)
    decision = recommend_algorithm(profile)
    result = run_selected_pipeline(values, decision)
    return {"profile": profile, "decision": decision, "result": result}


def _score(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {}
    return {
        "silhouette": round(float(silhouette_score(values, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(values, labels)), 4),
    }


def _validate_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("values must be a finite 2D array with at least 3 rows")
    return values
