import numpy as np

from src.hyperparameters.chromosome import ALPHA_MAX, ALPHA_MIN, random_alpha_chromosome


def test_alpha_chromosome_has_valid_length_and_range():
    chromosome = random_alpha_chromosome()
    assert chromosome.shape == (1,)
    assert ALPHA_MIN <= float(chromosome[0]) <= ALPHA_MAX
