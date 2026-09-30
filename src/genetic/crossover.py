"""Crossover operators for binary and real-coded GAs."""

from __future__ import annotations

import numpy as np


def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use a one-point crossover to generate two offspring."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length for crossover.")
    if len(parent1) < 2:
        raise ValueError("Chromosome must contain at least two genes.")
    cutoff = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:cutoff], parent2[cutoff:]))
    child2 = np.concatenate((parent2[:cutoff], parent1[cutoff:]))
    return child1.astype(parent1.dtype), child2.astype(parent2.dtype)


def arithmetic_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Create real-valued offspring by linear interpolation."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length for crossover.")
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2
