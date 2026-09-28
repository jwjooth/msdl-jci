"""Pytest fixtures: ensure src-layout import works with and without install."""

import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def sqlite_db(tmp_path_factory):
    """Seed all seven entity schemas with deterministic, synthetic data."""
    db_path = tmp_path_factory.mktemp("database") / "test.db"
    dates = pd.bdate_range("2020-01-01", periods=600)
    steps = np.arange(len(dates))
    close = 100 + steps * 0.05 + 5 * np.sin(steps / 10)
    prices = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Close": close,
            "High": close + 2,
            "Low": close - 2,
            "Open": close - 0.5,
            "Volume": 1_000_000 + steps * 100,
        }
    )
    periods = pd.date_range("2019-01-01", periods=48, freq="MS")
    months = [
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember",
    ]
    labels = [f"{months[d.month - 1]} {d.year}" for d in periods]
    with sqlite3.connect(db_path) as con:
        prices.to_sql("jci_historical", con, index=False)
        prices.drop(columns="Volume").to_sql("kurs_usdidr", con, index=False)
        pd.DataFrame(
            {
                "Period": [f"01 {label}" for label in labels],
                "BI-7Day-RR": [f"{5 + i % 6 / 4}%" for i in range(len(periods))],
            }
        ).to_sql("bi_rate", con, index=False)
        pd.DataFrame(
            {
                "Periode": labels,
                "Data Inflasi": [f"{2 + i % 8 / 10}%" for i in range(len(periods))],
            }
        ).to_sql("inflation_data", con, index=False)
        article = {
            "title": "Synthetic article",
            "url": "https://example.test/article",
            "scraped_at": "2020-01-01",
        }
        pd.DataFrame([{**article, "publish_date": "2020-01-01", "content": "Fixture"}]).to_sql(
            "cnbc_ihsg_articles", con, index=False
        )
        snippet = {**article, "published": "2020-01-01", "snippet": "Fixture"}
        pd.DataFrame([snippet]).to_sql("detik_ihsg_articles", con, index=False)
        pd.DataFrame([{**snippet, "category": "Markets"}]).to_sql(
            "kontan_ihsg_articles", con, index=False
        )
    return db_path


@pytest.fixture(scope="session", autouse=True)
def use_test_database(sqlite_db):
    """Route every cached settings consumer to the seeded DB before tests run."""
    from msdl_jci.config import settings

    test_settings = replace(settings.get_settings(), SQLITE_DB_PATH=sqlite_db)
    settings.get_settings.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "Settings", lambda: test_settings)
        try:
            yield
        finally:
            settings.get_settings.cache_clear()
