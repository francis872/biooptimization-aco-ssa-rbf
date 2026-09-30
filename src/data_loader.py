"""Utilities for loading and validating the SDSS dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import REQUIRED_COLUMNS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sdss_sample.csv"


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the SDSS CSV and validate its required schema."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'.")

    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    validate_numeric_columns(df)
    detect_nulls(df)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure all required dataset columns are present."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def validate_numeric_columns(df: pd.DataFrame) -> None:
    """Check that all expected numeric columns are numeric-like."""
    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "class"]
    bad_columns = []
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            bad_columns.append(column)
    if bad_columns:
        raise TypeError(f"Columns must be numeric: {bad_columns}")


def detect_nulls(df: pd.DataFrame) -> None:
    """Raise a clear error if null or NaN values exist."""
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Dataset contains null values in: {null_columns}")


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a lightweight summary for reporting."""
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": int(df.isnull().sum().sum()),
        "class_counts": df["class"].value_counts().to_dict(),
    }
    return summary
