import numpy as np
import pandas as pd

from ecommerce_clustering.phase2 import encode_review_topics, fuse_embeddings
from ecommerce_clustering.phase3 import cluster_embeddings
from ecommerce_clustering.phase4 import population_stability_index
from ecommerce_clustering.phase5 import select_configuration
from ecommerce_clustering.phase6 import build_segment_profiles


def test_phase2_to_phase6_components() -> None:
    values = np.vstack([np.random.default_rng(42).normal(loc, 0.2, (10, 3)) for loc in (0, 5, 10)])
    topics, _, _ = encode_review_topics(pd.Series(["great product"] * 30), n_topics=2)
    fused = fuse_embeddings(values, topics)
    clustered = cluster_embeddings(fused, n_clusters=3)
    selected = select_configuration(fused, cluster_options=(2, 3))
    profiles = build_segment_profiles(
        pd.DataFrame({"cluster": clustered["labels"], "recency": 10, "frequency": 2, "monetary": 50})
    )

    assert topics.shape == (30, 2)
    assert clustered["labels"].shape == (30,)
    assert selected["n_clusters"] in (2, 3)
    assert len(profiles) == 3
    assert population_stability_index(np.arange(20), np.arange(20) + 1) >= 0
