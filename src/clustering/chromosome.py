"""Chromosome representation for evolutionary clustering."""

from __future__ import annotations

import numpy as np


def random_centroid_chromosome(feature_matrix: np.ndarray, num_centroids: int = 3) -> np.ndarray:
    """Initialize centroid positions by sampling real rows from the dataset."""
    if feature_matrix.ndim != 2:
        raise ValueError("Feature matrix must be 2D.")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(feature_matrix), size=num_centroids, replace=False)
    return feature_matrix[indices].reshape(-1).astype(float)


def validate_clustering_chromosome(chromosome: np.ndarray, feature_count: int = 5, num_centroids: int = 3) -> bool:
    """Check that the chromosome has the expected real-valued layout."""
    expected_length = feature_count * num_centroids
    if chromosome.shape != (expected_length,):
        return False
    return np.isfinite(chromosome).all()


def clip_centroids(chromosome: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Clip chromosomes to the valid data range for each coordinate."""
    clipped = chromosome.copy().astype(float)
    reshaped = clipped.reshape(-1, len(mins))
    clipped_values = np.clip(reshaped, mins, maxs)
    return clipped_values.reshape(-1)
