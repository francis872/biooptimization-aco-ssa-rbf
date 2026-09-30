"""Utility functions shared by the genetic algorithms."""

from __future__ import annotations

import numpy as np


def random_individual(length: int, low: int = 0, high: int = 1) -> np.ndarray:
    """Generate a random integer-coded individual within the given bounds."""
    return np.random.randint(low, high + 1, size=length)


def evaluate_diversity(population: np.ndarray) -> float:
    """Estimate diversity as the average pairwise Hamming distance for binary populations."""
    if len(population) < 2:
        return 0.0
    all_distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            all_distances.append(np.mean(population[i] != population[j]))
    if not all_distances:
        return 0.0
    return float(np.mean(all_distances))
