"""SHAP-based explanations for cluster membership."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def explain_cluster_membership(
    model,
    values: np.ndarray,
    feature_names: list[str],
    cluster: int | None = None,
    max_samples: int = 50,
    background_size: int = 25,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Explain distance-based membership for one cluster.

    SHAP values describe which features move a customer closer to or farther
    from the selected cluster. A negative distance is used so larger output
    means stronger membership.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("values and feature_names must have matching dimensions")
    if len(values) == 0:
        raise ValueError("values must contain at least one row")
    selected_cluster = int(model.predict(values[:1])[0]) if cluster is None else int(cluster)
    if selected_cluster < 0 or selected_cluster >= model.n_clusters:
        raise ValueError("cluster is outside the model's cluster range")

    background = values[: min(background_size, len(values))]
    samples = values[: min(max_samples, len(values))]
    predict_membership = lambda rows: -model.transform(np.asarray(rows))[:, selected_cluster]
    explainer = shap.Explainer(predict_membership, background, algorithm="permutation")
    shap_values = np.asarray(explainer(samples).values)
    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return summary, shap_values
