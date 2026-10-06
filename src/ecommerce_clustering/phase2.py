"""Phase 2 lightweight representation learning components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class BehavioralAutoencoder:
    """An MLP autoencoder using the installed scikit-learn stack."""

    latent_dim: int = 8
    random_state: int = 42

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        self.scaler = StandardScaler().fit(values)
        scaled = self.scaler.transform(values)
        hidden = max(self.latent_dim * 2, 8)
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden, self.latent_dim, hidden),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=self.random_state,
        )
        self.model.fit(scaled, scaled)
        return self.transform(values)

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(values, dtype=float))
        activation = scaled
        for layer, bias in zip(self.model.coefs_[:2], self.model.intercepts_[:2]):
            activation = np.maximum(activation @ layer + bias, 0)
        return activation


def encode_review_topics(
    reviews: pd.Series,
    n_topics: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, CountVectorizer, LatentDirichletAllocation]:
    """Encode review text as interpretable LDA topic distributions."""
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    counts = vectorizer.fit_transform(reviews.fillna("").astype(str))
    if counts.shape[1] == 0:
        return np.zeros((len(reviews), n_topics)), vectorizer, LatentDirichletAllocation()
    topics = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_state, learning_method="batch"
    )
    return topics.fit_transform(counts), vectorizer, topics


def fuse_embeddings(*embeddings: np.ndarray) -> np.ndarray:
    """Concatenate and standardize multiple customer representations."""
    arrays = [np.asarray(embedding, dtype=float) for embedding in embeddings]
    if not arrays or len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("Embeddings must be non-empty and have the same row count")
    return StandardScaler().fit_transform(np.concatenate(arrays, axis=1))
