"""Fitness functions for the feature-selection GA."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


def feature_selection_fitness(chromosome: np.ndarray, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate accuracy using selected features only."""
    selected = np.where(chromosome == 1)[0]
    if selected.size == 0:
        raise ValueError("A chromosome with no active features is invalid.")
    model = KNeighborsClassifier(n_neighbors=5)
    X_train_selected = X_train[:, selected]
    X_test_selected = X_test[:, selected]
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    return float(np.mean(predictions == y_test))
