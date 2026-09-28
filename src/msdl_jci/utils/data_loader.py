"""CSV and SQLite data loading helpers."""

import sqlite3
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


def read_sqlite_table(table_name: str, db_path: str | Path) -> pd.DataFrame:
    """Read a whole SQLite table into a DataFrame, validated against ENTITY_TABLES."""
    # ponytail: stdlib sqlite3, not SQLAlchemy — single-file DB, no server, no ORM gain.
    # Upgrade path: sqlalchemy.create_engine if this ever needs pools/concurrency.
    with sqlite3.connect(Path(db_path)) as con:
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", con)
    expected = ENTITY_TABLES.get(table_name)
    if expected is not None:
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
    """Read a numeric series from a SQLite table with automatic symbol stripping."""
    df = read_sqlite_table(table_name, db_path)

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found. Available: {list(df.columns)}")

    return _clean_numeric(df[column_name], remove_symbol)
