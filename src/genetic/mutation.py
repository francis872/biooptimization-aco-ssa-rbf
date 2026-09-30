"""Mutation operators for binary and real-coded GAs."""

from __future__ import annotations

import numpy as np


def bit_flip_mutation(chromosome: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Flip each bit with the given mutation probability."""
    mutated = chromosome.copy()
    for idx in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[idx] = 1 - mutated[idx]
    return mutated


def gaussian_mutation(chromosome: np.ndarray, mutation_rate: float = 0.1, sigma: float = 0.1) -> np.ndarray:
    """Apply Gaussian perturbation to each gene with some probability."""
    mutated = chromosome.copy()
    for idx in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[idx] += np.random.normal(0, sigma)
    return mutated
