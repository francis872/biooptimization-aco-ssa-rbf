import numpy as np

from src.feature_selection.chromosome import random_feature_chromosome, repair_feature_chromosome


def test_feature_chromosome_length_is_six():
    chromosome = random_feature_chromosome()
    assert chromosome.shape == (6,)
    assert set(np.unique(chromosome)).issubset({0, 1})


def test_feature_chromosome_has_at_least_one_active_gene():
    chromosome = np.zeros(6, dtype=int)
    repaired = repair_feature_chromosome(chromosome)
    assert repaired.shape == (6,)
    assert np.sum(repaired) >= 1
