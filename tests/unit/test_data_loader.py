"""Unit tests for data loader."""

import numpy as np
import pytest

from msdl_jci.utils.data_loader import read_numeric_series


def test_read_numeric_series_success(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("Date,Value\n2023-01-01,10.5%\n2023-01-02,11.0%\n")

    res = read_numeric_series(csv_file, "Value", remove_symbol="%")
    assert isinstance(res, np.ndarray)
    assert len(res) == 2
    assert np.isclose(res[0], 10.5)
    assert np.isclose(res[1], 11.0)


def test_read_numeric_series_missing_column(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("Date,Price\n2023-01-01,100\n")

    with pytest.raises(KeyError):
        read_numeric_series(csv_file, "NonExistentColumn")


def test_read_numeric_series_missing_file():
    with pytest.raises(FileNotFoundError):
        read_numeric_series("non_existent_file_path.csv", "Price")
