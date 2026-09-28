"""Unit tests for data loader."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from msdl_jci.utils.data_loader import (
    read_numeric_series,
    read_sqlite_series,
    read_sqlite_table,
)


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


@pytest.mark.parametrize("table_name", ["unknown", "bi_rate] WHERE 1=1 --"])
def test_read_sqlite_table_rejects_unknown_before_connecting(tmp_path, table_name):
    db = tmp_path / "must_not_be_created.db"
    with pytest.raises(ValueError, match="Unknown table"):
        read_sqlite_table(table_name, db)
    assert not db.exists()


def test_read_sqlite_table_rejects_unregistered_existing_table(tmp_path):
    db = tmp_path / "sample.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE unregistered (value REAL)")
    with pytest.raises(ValueError, match="Unknown table"):
        read_sqlite_table("unregistered", db)


def test_read_sqlite_table_checks_recognized_schema(tmp_path):
    db = tmp_path / "sample.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE bi_rate (Period TEXT)")
    with pytest.raises(ValueError, match="missing columns.*BI-7Day-RR"):
        read_sqlite_table("bi_rate", db)


@pytest.mark.parametrize(
    "table,date_column,value_column,dates,values,symbol",
    [
        (
            "bi_rate",
            "Period",
            "BI-7Day-RR",
            ["02 Januari 2024", "15 Desember 2023", "01 Januari 2024"],
            ["7%", "9%", "4%"],
            "%",
        ),
        (
            "inflation_data",
            "Periode",
            "Data Inflasi",
            ["Februari 2024", "Desember 2023", "Januari 2024"],
            ["7%", "9%", "4%"],
            "%",
        ),
        ("kurs_usdidr", "Date", "Close", ["2024-01-02", "2023-12-15", "2024-01-01"], [7, 9, 4], ""),
    ],
)
def test_read_sqlite_series_sorts_dates_with_values(
    tmp_path, table, date_column, value_column, dates, values, symbol
):
    db = tmp_path / "macro.db"
    frame = pd.DataFrame({date_column: dates, value_column: values})
    if table == "kurs_usdidr":
        for col in ("High", "Low", "Open"):
            frame[col] = values
    with sqlite3.connect(db) as con:
        frame.to_sql(table, con, index=False)

    result = read_sqlite_series(table, value_column, db, symbol)

    np.testing.assert_array_equal(result, [9.0, 4.0, 7.0])
    assert result.dtype == np.float64
