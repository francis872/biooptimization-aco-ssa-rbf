"""Fitness for Ridge alpha optimization."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def alpha_fitness(alpha: float, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
    """Return (fitness, mse, r2) for a Ridge alpha candidate."""
    model = Ridge(alpha=float(alpha))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = float(mean_squared_error(y_test, predictions))
    epsilon = 1e-8
    fitness = 1.0 / (mse + epsilon)
    r2 = float(model.score(X_test, y_test))
    return fitness, mse, r2
