"""Fitness functions for evolutionary clustering."""

from __future__ import annotations

import numpy as np


def assign_clusters(X: np.ndarray, chromosome: np.ndarray, num_centroids: int = 3) -> np.ndarray:
    """Assign each row to the nearest centroid."""
    centroids = chromosome.reshape(num_centroids, -1)
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def clustering_sse(X: np.ndarray, chromosome: np.ndarray, num_centroids: int = 3) -> float:
    """Compute the within-cluster sum of squared errors."""
    labels = assign_clusters(X, chromosome, num_centroids)
    centroids = chromosome.reshape(num_centroids, -1)
    sse = 0.0
    for center_id in range(num_centroids):
        cluster_points = X[labels == center_id]
        if len(cluster_points) == 0:
            continue
        diff = cluster_points - centroids[center_id]
        sse += float(np.sum(diff ** 2))
    return sse


def clustering_fitness(X: np.ndarray, chromosome: np.ndarray, epsilon: float = 1e-8, num_centroids: int = 3) -> float:
    """Return 1/(SSE + epsilon), higher is better."""
    sse = clustering_sse(X, chromosome, num_centroids=num_centroids)
    return 1.0 / (sse + epsilon)
