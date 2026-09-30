"""Selection operators for genetic algorithms."""

from __future__ import annotations

import numpy as np

from src.config import RANDOM_STATE, TOURNAMENT_SIZE


def tournament_selection(population: np.ndarray, fitness: np.ndarray, tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Return the winner from a tournament with replacement."""
    if population.size == 0:
        raise ValueError("Population cannot be empty.")
    if tournament_size <= 0:
        raise ValueError("Tournament size must be positive.")
    indices = np.random.default_rng(RANDOM_STATE).choice(len(population), size=tournament_size, replace=False)
    best_index = indices[np.argmax(fitness[indices])]
    return population[best_index].copy()


def roulette_wheel_selection(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """Select an individual based on proportional fitness."""
    if population.size == 0:
        raise ValueError("Population cannot be empty.")
    if np.any(fitness < 0):
        fitness = fitness - np.min(fitness) + 1e-8
    total = np.sum(fitness)
    if total <= 0:
        raise ValueError("Total fitness must be positive.")
    probs = fitness / total
    idx = np.random.default_rng(RANDOM_STATE).choice(len(population), p=probs)
    return population[idx].copy()
