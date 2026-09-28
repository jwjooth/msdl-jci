"""CSV and SQLite data loading helpers."""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Entity contract: verbatim schema of database/main_database.db. Every SQLite
# read in the project funnels through read_sqlite_table, which enforces this.
ENTITY_TABLES: dict[str, tuple[str, ...]] = {
    "jci_historical": ("Date", "Close", "High", "Low", "Open", "Volume"),
    "bi_rate": ("Period", "BI-7Day-RR"),
    "inflation_data": ("Periode", "Data Inflasi"),
    "kurs_usdidr": ("Date", "Close", "High", "Low", "Open"),
    "cnbc_ihsg_articles": ("title", "url", "publish_date", "content", "scraped_at"),
    "detik_ihsg_articles": ("title", "url", "snippet", "published", "scraped_at"),
    "kontan_ihsg_articles": ("title", "url", "snippet", "published", "category", "scraped_at"),
}


MONTH_MAP = {
    "januari": "01",
    "jan": "01",
    "februari": "02",
    "feb": "02",
    "maret": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "mei": "05",
    "may": "05",
    "juni": "06",
    "jun": "06",
    "juli": "07",
    "jul": "07",
    "agustus": "08",
    "agu": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "oktober": "10",
    "okt": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "desember": "12",
    "des": "12",
    "dec": "12",
}


def parse_indonesian_date(series: pd.Series, day_first: bool = True) -> pd.Series:
    """Parse Indonesian text date strings into pandas datetime series."""
    s = series.astype(str).str.strip().str.lower()
    for id_month, num_month in MONTH_MAP.items():
        s = s.str.replace(id_month, num_month, regex=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", dayfirst=day_first)


def read_sqlite_table(table_name: str, db_path: str | Path) -> pd.DataFrame:
    """Read a whole SQLite table into a DataFrame, validated against ENTITY_TABLES."""
    # ponytail: stdlib sqlite3, not SQLAlchemy — single-file DB, no server, no ORM gain.
    # Upgrade path: sqlalchemy.create_engine if this ever needs pools/concurrency.
    if table_name not in ENTITY_TABLES:
        raise ValueError(f"Unknown table: {table_name!r}")
    expected = ENTITY_TABLES[table_name]
    with sqlite3.connect(Path(db_path)) as con:
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", con)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Table [{table_name}] missing columns {missing}: {list(df.columns)}")
    return df


def load_frame(
    source: str | Path | pd.DataFrame | None,
    table_name: str,
    db_path: str | Path,
) -> pd.DataFrame:
    """Load a DataFrame from an explicit CSV/frame, else the SQLite table."""
    if source is None:
        return read_sqlite_table(table_name, db_path)
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(Path(source))


def _clean_numeric(series: pd.Series, remove_symbol: str = "") -> NDArray[np.float64]:
    """Convert a series to float64 values, raising on invalid numeric data.

    When ``remove_symbol`` is set, strip it, commas, and surrounding whitespace
    before conversion.
    """
    if remove_symbol:
        series = (
            series.astype(str)
            .str.replace(remove_symbol, "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)


def read_numeric_series(
    csv_path: str | Path,
    column_name: str,
    remove_symbol: str = "",
) -> NDArray[np.float64]:
    """Read a numeric series from a CSV file with automatic symbol stripping."""
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found at path: {path_obj}")

    df = pd.read_csv(path_obj)

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found. Available: {list(df.columns)}")

    return _clean_numeric(df[column_name], remove_symbol)


def read_sqlite_series(
    table_name: str,
    column_name: str,
    db_path: str | Path,
    remove_symbol: str = "",
) -> NDArray[np.float64]:
    """Read a numeric series, sorting macro tables by their parsed dates."""
    df = read_sqlite_table(table_name, db_path)
    date_column = {"bi_rate": "Period", "inflation_data": "Periode", "kurs_usdidr": "Date"}.get(
        table_name
    )
    if date_column is not None:
        if table_name == "kurs_usdidr":
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        else:
            df[date_column] = parse_indonesian_date(
                df[date_column], day_first=table_name == "bi_rate"
            )
        df = df.sort_values(date_column, kind="stable")

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found. Available: {list(df.columns)}")

    return _clean_numeric(df[column_name], remove_symbol)
