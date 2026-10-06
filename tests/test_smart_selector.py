import numpy as np

from ecommerce_clustering.smart_selector import profile_dataset, recommend_algorithm, smart_cluster


def test_small_dataset_selects_customer_preserving_pipeline() -> None:
    values = np.random.default_rng(42).normal(size=(30, 3))
    decision = recommend_algorithm(profile_dataset(values))
    assert decision["algorithm"] in {"kmeans", "kmeans_flag_outliers"}


def test_high_dimensional_data_selects_autoencoder_pipeline() -> None:
    values = np.random.default_rng(42).normal(size=(200, 15))
    assert recommend_algorithm(profile_dataset(values))["algorithm"] == "autoencoder_kmeans"


def test_large_outlier_data_can_select_hybrid() -> None:
    values = np.vstack(
        [np.random.default_rng(42).normal(0, 0.1, (1000, 3)), np.random.default_rng(43).normal(5, 0.1, (50, 3))]
    )
    decision = recommend_algorithm(profile_dataset(values))
    assert decision["algorithm"] in {"hybrid_dbscan_kmeans", "kmeans"}


def test_smart_cluster_returns_profile_decision_and_result() -> None:
    values = np.vstack(
        [np.random.default_rng(42).normal(0, 0.2, (20, 3)), np.random.default_rng(43).normal(4, 0.2, (20, 3))]
    )
    output = smart_cluster(values)
    assert output["decision"]["algorithm"]
    assert "metrics" in output["result"]
