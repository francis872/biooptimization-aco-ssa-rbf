"""Chromosome representation for alpha optimization."""

from __future__ import annotations

import numpy as np

ALPHA_MIN = 1e-4
ALPHA_MAX = 100.0


def validate_alpha_chromosome(chromosome: np.ndarray) -> bool:
    """Ensure a real-valued chromosome stays within the alpha bounds."""
    if not isinstance(chromosome, np.ndarray):
        return False
    if chromosome.ndim != 1 or chromosome.size != 1:
        return False
    value = float(chromosome[0])
    return ALPHA_MIN <= value <= ALPHA_MAX


def clip_alpha(value: float) -> float:
    """Keep alpha in a valid range after mutation or crossover."""
    return float(np.clip(value, ALPHA_MIN, ALPHA_MAX))


def random_alpha_chromosome() -> np.ndarray:
    """Generate a valid random chromosome for alpha."""
    return np.array([float(np.random.uniform(ALPHA_MIN, ALPHA_MAX))], dtype=float)
