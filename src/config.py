"""Global configuration for the SDSS genetic algorithms project."""

from __future__ import annotations

import random

import numpy as np

RANDOM_STATE = 42
POPULATION_SIZE = 40
GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
ELITISM = 1

FEATURE_COLUMNS = ["u", "g", "r", "i", "z", "redshift"]
TARGET_COLUMN = "class"
REGRESSION_TARGET = "redshift"
CLUSTER_COLUMNS = ["u", "g", "r", "i", "z"]
REQUIRED_COLUMNS = ["u", "g", "r", "i", "z", "redshift", "class"]


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

try:
    import sklearn

    sklearn.utils.check_random_state(RANDOM_STATE)
except Exception:
    pass
