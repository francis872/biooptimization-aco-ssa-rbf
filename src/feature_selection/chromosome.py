"""Chromosome representation for feature selection."""

from __future__ import annotations

import numpy as np

FEATURE_COLUMNS = ["u", "g", "r", "i", "z", "redshift"]


def validate_feature_chromosome(chromosome: np.ndarray) -> bool:
    """Check binary chromosome validity."""
    if len(chromosome) != len(FEATURE_COLUMNS) if False else False:
        return False
    if chromosome.shape != (len(FEATURE_COLUMNS),):
        return False
    if not np.isin(chromosome, [0, 1]).all():
        return False
    if np.sum(chromosome) == 0:
        return False
    return True


def repair_feature_chromosome(chromosome: np.ndarray) -> np.ndarray:
    """Ensure at least one feature is selected."""
    repaired = np.asarray(chromosome, dtype=int).copy()
    if repaired.size != len(FEATURE_COLUMNS):
        raise ValueError(f"Chromosome length must be {len(FEATURE_COLUMNS)}.")
    if np.all(repaired == 0):
        repaired[np.random.randint(0, len(FEATURE_COLUMNS))] = 1
    return repaired


def random_feature_chromosome() -> np.ndarray:
    """Generate a valid random binary chromosome."""
    chromosome = np.random.randint(0, 2, size=len(FEATURE_COLUMNS), dtype=int)
    return repair_feature_chromosome(chromosome)
