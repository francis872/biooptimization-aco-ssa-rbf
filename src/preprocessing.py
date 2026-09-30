"""Reusable preprocessing utilities with train/test leakage prevention."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import RANDOM_STATE, TARGET_COLUMN


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train/test sets while preserving labels."""
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return X_train, X_test, y_train, y_test


def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode a classification target and return the fitted encoder."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)
    return encoded, encoder


def fit_scaler(X_train: pd.DataFrame | np.ndarray) -> StandardScaler:
    """Fit a scaler only on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(
    X: pd.DataFrame | np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply the fitted scaler to data without refitting."""
    return scaler.transform(X)


def prepare_feature_dataset(df: pd.DataFrame, feature_columns: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Select the features and target while preserving column order."""
    features = df[list(feature_columns)].copy()
    target = df[TARGET_COLUMN].copy()
    return features, target


def safe_float_array(values: Iterable[float]) -> np.ndarray:
    """Convert values to a NumPy array, rejecting NaNs and infinities."""
    arr = np.asarray(list(values), dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Feature values contain NaN or infinite numbers.")
    return arr
