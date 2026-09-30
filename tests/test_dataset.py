import pandas as pd

from src.data_loader import load_dataset


def test_dataset_file_exists():
    dataset = load_dataset("data/sdss_sample.csv")
    assert isinstance(dataset, pd.DataFrame)
    assert not dataset.empty


def test_dataset_contains_required_columns():
    dataset = load_dataset("data/sdss_sample.csv")
    required = ["u", "g", "r", "i", "z", "redshift", "class"]
    for column in required:
        assert column in dataset.columns


def test_dataset_numeric_columns_are_valid():
    dataset = load_dataset("data/sdss_sample.csv")
    numeric_columns = ["u", "g", "r", "i", "z", "redshift"]
    for column in numeric_columns:
        assert pd.api.types.is_numeric_dtype(dataset[column])
