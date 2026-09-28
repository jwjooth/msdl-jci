"""Unit tests for PR #15's notebook utilities, without running its demo or plots.

Uses unittest from the standard library and the notebook's declared dependencies.
Run from the repository root with: python -m unittest discover -s tests -v
All inputs are synthetic; local .env files and the research database are isolated.
"""

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "development.ipynb"
# Independent fixtures for the external SQLite contract, including column order.
TABLE_COLUMNS = {
    "jci_historical": ("Date", "Close", "High", "Low", "Open", "Volume"),
    "bi_rate": ("Period", "BI-7Day-RR"),
    "inflation_data": ("Periode", "Data Inflasi"),
    "kurs_usdidr": ("Date", "Close", "High", "Low", "Open"),
    "cnbc_ihsg_articles": ("title", "url", "publish_date", "content", "scraped_at"),
    "detik_ihsg_articles": ("title", "url", "snippet", "published", "scraped_at"),
    "kontan_ihsg_articles": ("title", "url", "snippet", "published", "category", "scraped_at"),
}


def load_notebook(root, env=None):
    """Execute complete definition cells in a fresh namespace, not copied utilities."""
    cells = {cell["id"]: cell for cell in json.loads(NOTEBOOK.read_text())["cells"]}
    module = ModuleType("_notebook_under_test")
    with (
        patch.dict(sys.modules, {module.__name__: module}),
        patch.dict(os.environ, env or {}, clear=True),
        patch("pathlib.Path.cwd", return_value=root / "notebooks"),
        patch("dotenv.load_dotenv") as dotenv_loader,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        for cell_id in ("imports", "config", "entity", "utils"):
            source = "".join(cells[cell_id]["source"])
            # Only the display magic is notebook-specific; all Python stays intact.
            source = source.replace("%matplotlib inline\n", "")
            # Import code from the repository notebook, which has no Python module.
            exec(compile(source, f"{NOTEBOOK}#{cell_id}", "exec"), module.__dict__)  # noqa: S102
    module.dotenv_loader = dotenv_loader
    return module


def prices(n=60, close=None):
    close = np.asarray(close if close is not None else 100 + np.arange(n), dtype=float)
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
        "Open": close, "Close": close, "High": close + 1, "Low": close - 1,
        "Volume": np.full(len(close), 1000.0),
    })


def raw_frames(n=60, close=None):
    jci = prices(n, close)
    bi = pd.DataFrame({"Period": ["1 Januari 2024"], "BI-7Day-RR": [" 5.5 % "]})
    inf = pd.DataFrame({"Periode": ["Januari 2024"], "Data Inflasi": [" 2.5 % "]})
    kur = pd.DataFrame({"Date": jci["Date"], "Close": "15,000"})
    return jci, bi, inf, kur


def tensor_frame():
    """Distinct feature offsets and endpoint values expose column/row swaps."""
    values = np.array([0, 2, 4, 6, 20, -10], dtype=float)
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=6, freq="B"),
        "Close": 100 + values, "Volume": 200 + 2 * values,
        "bi_rate": 5 + values, "inflation_rate": 10 + 3 * values,
        "emb_0": [10, 11, 12, 13, 14, 15], "emb_1": [-10, -11, -12, -13, -14, -15],
        "target_direction": [0, 1, 1, 0, 1, 0],
        "future_close": 99999.0, "return_5d": -99999.0,
    })


class NotebookTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "pyproject.toml").touch()
        (self.root / "notebooks").mkdir()
        self.nb = load_notebook(self.root)


class ConfigTests(NotebookTestCase):
    def test_defaults_and_root_resolution_from_notebooks(self):
        cfg = self.nb.CFG
        self.assertEqual((cfg.look_back, cfg.horizon, cfg.seed, cfg.news_dim), (28, 5, 42, 768))
        self.assertEqual(cfg.tech_cols,
                         ("Close", "Volume", "RSI_14", "MACD", "MACD_Signal", "ATR_14", "SMA_20"))
        self.assertEqual(cfg.macro_cols, ("bi_rate", "inflation_rate", "usd_idr"))
        self.assertEqual(self.nb.NB_ROOT, self.root)
        self.assertEqual(self.nb.DB_PATH, self.root / "database" / "main_database.db")
        self.nb.dotenv_loader.assert_called_once_with(self.root / ".env")
        self.assertFalse(self.nb.USE_DB)
        with self.assertRaises(FrozenInstanceError):
            cfg.horizon = 10

    def test_environment_overrides_and_relative_database_path(self):
        nb = load_notebook(self.root, {
            "DATABASE_PATH": "custom/research.db", "ML_LOOK_BACK": "7",
            "ML_PREDICTION_HORIZON": "2", "ML_RANDOM_STATE": "123",
        })
        self.assertEqual((nb.CFG.look_back, nb.CFG.horizon, nb.CFG.seed), (7, 2, 123))
        self.assertEqual(nb.DB_PATH, self.root / "custom" / "research.db")

    def test_absolute_existing_database_selects_sqlite(self):
        db = self.root / "existing.db"
        with sqlite3.connect(db):
            pass
        nb = load_notebook(self.root, {"DATABASE_PATH": str(db)})
        self.assertEqual(nb.DB_PATH, db)
        self.assertTrue(nb.USE_DB)

    def test_invalid_integer_environment_values_fail_early(self):
        for name in ("ML_LOOK_BACK", "ML_PREDICTION_HORIZON", "ML_RANDOM_STATE"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_notebook(self.root, {name: "not-an-integer"})


class SQLiteTests(NotebookTestCase):
    def setUp(self):
        super().setUp()
        self.nb.DB_PATH = self.root / "fixture.db"

    def write_table(self, name, columns, rows=1):
        frame = pd.DataFrame({column: [f"{column}-{i}" for i in range(rows)] for column in columns})
        with sqlite3.connect(self.nb.DB_PATH) as con:
            frame.to_sql(name, con, index=False, if_exists="replace")
        return frame

    def test_every_entity_reads_values_and_allows_extra_columns(self):
        self.assertEqual(self.nb.ENTITY_TABLES, TABLE_COLUMNS)
        for name, columns in TABLE_COLUMNS.items():
            with self.subTest(table=name):
                expected = self.write_table(name, (*columns, "extra_metadata"), rows=2)
                assert_frame_equal(self.nb.read_table(name), expected)

    def test_each_required_column_is_checked_on_every_read(self):
        for name, columns in TABLE_COLUMNS.items():
            for missing in columns:
                with self.subTest(table=name, missing=missing):
                    self.write_table(name, columns)
                    self.nb.read_table(name)
                    self.write_table(name, [column for column in columns if column != missing])
                    with self.assertRaises(ValueError) as caught:
                        self.nb.read_table(name)
                    self.assertIn(f"Table [{name}] missing columns", str(caught.exception))
                    self.assertIn(repr(missing), str(caught.exception))

    def test_valid_empty_tables_remain_empty(self):
        for name, columns in TABLE_COLUMNS.items():
            with self.subTest(table=name):
                self.write_table(name, columns, rows=0)
                result = self.nb.read_table(name)
                self.assertTrue(result.empty)
                self.assertEqual(tuple(result.columns), columns)

    def test_missing_table_raises_database_error(self):
        with self.assertRaisesRegex(pd.errors.DatabaseError, "no such table"):
            self.nb.read_table("jci_historical")

    def test_load_frames_reads_the_four_market_tables_in_order(self):
        names = ("jci_historical", "bi_rate", "inflation_data", "kurs_usdidr")
        expected = [self.write_table(name, TABLE_COLUMNS[name]) for name in names]
        self.nb.USE_DB = True
        frames = self.nb.load_frames()
        self.assertEqual(len(frames), 4)
        for actual, wanted in zip(frames, expected, strict=True):
            assert_frame_equal(actual, wanted)

    def test_load_frames_propagates_schema_errors(self):
        self.nb.USE_DB = True
        self.write_table("jci_historical", ("Date", "Close"))
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.nb.load_frames()


class SyntheticDataTests(NotebookTestCase):
    def test_fallback_is_repeatable_and_does_not_read_sqlite(self):
        with patch.object(self.nb, "read_table", side_effect=AssertionError("unexpected DB read")):
            first = self.nb.load_frames()
            second = self.nb.load_frames()
        self.assertEqual([len(frame) for frame in first], [500, 17, 17, 500])
        for actual, expected in zip(first, second, strict=True):
            assert_frame_equal(actual, expected)
        jci, _, _, kur = first
        assert_series_equal(jci["Date"], kur["Date"])
        self.assertTrue(jci["Date"].is_monotonic_increasing)
        self.assertTrue((jci["Date"].dt.dayofweek < 5).all())
        self.assertTrue((jci["High"] >= jci["Close"]).all())
        self.assertTrue((jci["Low"] <= jci["Close"]).all())
        self.assertFalse(self.nb.DB_PATH.exists())

    def test_seed_changes_both_price_series_but_not_calendar_or_macro(self):
        first = self.nb.load_frames()
        self.nb.CFG = replace(self.nb.CFG, seed=99)
        second = self.nb.load_frames()
        for index in (0, 3):
            self.assertFalse(first[index]["Close"].equals(second[index]["Close"]))
            assert_series_equal(first[index]["Date"], second[index]["Date"])
        for index in (1, 2):
            assert_frame_equal(first[index], second[index])


class DateParsingTests(NotebookTestCase):
    def test_indonesian_months_and_abbreviations(self):
        months = [
            ("Januari", "jan"), ("Februari", "feb"), ("Maret", "mar"), ("April", "apr"),
            ("Mei", "may"), ("Juni", "jun"), ("Juli", "jul"), ("Agustus", "agu", "aug"),
            ("September", "sep"), ("Oktober", "okt", "oct"), ("November", "nov"),
            ("Desember", "des", "dec"),
        ]
        for month, spellings in enumerate(months, start=1):
            for spelling in spellings:
                with self.subTest(month=month, spelling=spelling):
                    result = self.nb.parse_id_date(pd.Series([f" 15 {spelling.upper()} 2024 "]))
                    self.assertEqual(result.iloc[0], pd.Timestamp(2024, month, 15))

    def test_month_only_dates_and_day_first_option(self):
        result = self.nb.parse_id_date(pd.Series(["Februari 2024", "03/04/2024"]), day_first=False)
        self.assertEqual(result.tolist(), [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-04")])
        result = self.nb.parse_id_date(pd.Series(["03/04/2024"]), day_first=True)
        self.assertEqual(result.iloc[0], pd.Timestamp("2024-04-03"))

    def test_invalid_dates_are_coerced_and_input_index_is_preserved(self):
        source = pd.Series([None, "", "bad-date", "31 Februari 2024", "29 Februari 2024"],
                           index=[9, 4, 8, 3, 7], name="dates")
        before = source.copy()
        result = self.nb.parse_id_date(source)
        self.assertTrue(result.iloc[:4].isna().all())
        self.assertEqual(result.iloc[4], pd.Timestamp("2024-02-29"))
        self.assertEqual(result.index.tolist(), source.index.tolist())
        assert_series_equal(source, before)


class IndicatorTests(NotebookTestCase):
    def test_constant_prices_have_finite_indicators_after_warmup(self):
        result = self.nb.add_indicators(prices(close=np.full(30, 100)))
        for column, expected in {"RSI_14": 0, "MACD": 0, "MACD_Signal": 0,
                                 "ATR_14": 2, "SMA_20": 100}.items():
            with self.subTest(column=column):
                assert_allclose(result[column].iloc[19:], expected)

    def test_warmup_boundaries_and_trending_prices(self):
        result = self.nb.add_indicators(prices())
        for column, first in (("RSI_14", 14), ("ATR_14", 13), ("SMA_20", 19)):
            with self.subTest(column=column):
                self.assertTrue(result[column].iloc[:first].isna().all())
                self.assertTrue(result[column].iloc[first:].notna().all())
        self.assertAlmostEqual(result.loc[19, "SMA_20"], 109.5)
        self.assertAlmostEqual(result.loc[14, "RSI_14"], 100, places=5)
        falling = self.nb.add_indicators(prices(close=200 - np.arange(30)))
        assert_allclose(falling["RSI_14"].iloc[14:], 0)

    def test_atr_includes_overnight_gap_and_macd_tracks_a_price_jump(self):
        result = self.nb.add_indicators(prices(close=[100] * 20 + [120]))
        self.assertAlmostEqual(result.loc[20, "ATR_14"], 2 + 19 / 14)
        macd = 20 * (2 / 13 - 2 / 27)
        self.assertAlmostEqual(result.loc[20, "MACD"], macd)
        self.assertAlmostEqual(result.loc[20, "MACD_Signal"], macd * 2 / 10)

    def test_future_price_shocks_do_not_change_past_indicators(self):
        original = prices()
        changed = original.copy()
        changed.loc[40:, ["Close", "High", "Low"]] += 10000
        baseline = self.nb.add_indicators(original)
        shocked = self.nb.add_indicators(changed)
        assert_frame_equal(baseline.iloc[:40], shocked.iloc[:40])
        self.assertNotEqual(baseline.loc[40, "MACD"], shocked.loc[40, "MACD"])


class AlignmentTests(NotebookTestCase):
    def setUp(self):
        super().setUp()
        self.nb.CFG = replace(self.nb.CFG, news_dim=3)

    def test_labels_use_configured_trading_row_horizon_and_drop_unknown_tail(self):
        # Oscillation covers up, down, and equal closes across weekends.
        close = np.tile([100, 103, 98, 100, 105, 99, 100], 10)
        for horizon in (1, 5, 7):
            with self.subTest(horizon=horizon):
                self.nb.CFG = replace(self.nb.CFG, horizon=horizon)
                frames = raw_frames(close=close)
                result = self.nb.align(*frames)
                expected_dates = frames[0]["Date"].iloc[19:-horizon].tolist()
                self.assertEqual(result["Date"].tolist(), expected_dates)
                current, future = close[19:-horizon], close[19 + horizon:]
                assert_array_equal(result["future_close"], future)
                assert_array_equal(result["target_direction"], (future > current).astype(float))
                assert_allclose(result["return_5d"], (future - current) / current)
                if horizon == 7:
                    assert_array_equal(result["target_direction"], 0)

    def test_macro_availability_drops_leading_gaps_and_never_backfills(self):
        for delayed in ("bi_rate", "inflation_rate", "usd_idr"):
            with self.subTest(delayed=delayed):
                jci, bi, inf, kur = raw_frames()
                release = jci.loc[25, "Date"]
                if delayed == "bi_rate":
                    bi.loc[0, "Period"] = release.strftime("%d/%m/%Y")
                elif delayed == "inflation_rate":
                    inf.loc[0, "Periode"] = release.strftime("%m/%d/%Y")
                else:
                    kur = kur.iloc[25:].copy()
                result = self.nb.align(jci, bi, inf, kur)
                self.assertEqual(result["Date"].iloc[0], release)
                self.assertEqual(len(result), len(jci) - 25 - self.nb.CFG.horizon)
                self.assertFalse(result[list(self.nb.CFG.macro_cols)].isna().any().any())

    def test_unsorted_macro_releases_take_effect_on_release_day_and_forward_fill(self):
        jci, _, _, _ = raw_frames()
        bi = pd.DataFrame({"Period": ["12 Februari 2024", "1 Januari 2024"],
                           "BI-7Day-RR": ["6 %", "5 %"]})
        inf = pd.DataFrame({"Periode": ["Februari 2024", "Januari 2024"],
                            "Data Inflasi": ["3 %", "2 %"]})
        kur = pd.DataFrame({"Date": ["2024-02-12", "2024-01-01", "2024-02-13"],
                            "Close": ["16,000", "15,000", "invalid"]})
        result = self.nb.align(jci.iloc[::-1], bi, inf, kur).set_index("Date")
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertEqual(result.loc["2024-02-09", "bi_rate"], 5)
        self.assertEqual(result.loc["2024-02-12", "bi_rate"], 6)
        self.assertEqual(result.loc["2024-01-31", "inflation_rate"], 2)
        self.assertEqual(result.loc["2024-02-01", "inflation_rate"], 3)
        self.assertEqual(result.loc["2024-02-09", "usd_idr"], 15000)
        self.assertEqual(result.loc["2024-02-12", "usd_idr"], 16000)
        self.assertEqual(result.loc["2024-02-13", "usd_idr"], 16000)
        self.assertEqual(result.loc["2024-02-14", "usd_idr"], 16000)

    def test_invalid_dates_and_missing_closes_are_ignored_without_mutating_inputs(self):
        frames = list(raw_frames())
        frames[0] = pd.concat([frames[0], pd.DataFrame({"Date": ["bad", "2025-01-01"],
                                                       "Close": [500, np.nan]})], ignore_index=True)
        for index, date_column, value_column in ((1, "Period", "BI-7Day-RR"),
                                                  (2, "Periode", "Data Inflasi"),
                                                  (3, "Date", "Close")):
            frames[index] = pd.concat([frames[index], pd.DataFrame({date_column: ["bad"],
                                                                  value_column: ["99"]})],
                                      ignore_index=True)
        before = [frame.copy(deep=True) for frame in frames]
        expected = self.nb.align(*raw_frames())
        assert_frame_equal(self.nb.align(*frames), expected)
        for actual, original in zip(frames, before, strict=True):
            assert_frame_equal(actual, original)

    def test_news_fallback_has_exact_configured_width_and_only_zeros(self):
        self.nb.CFG = replace(self.nb.CFG, news_dim=768)
        result = self.nb.align(*raw_frames())
        embeddings = result.filter(regex=r"^emb_")
        self.assertEqual(embeddings.columns.tolist(), [f"emb_{i}" for i in range(768)])
        self.assertEqual(embeddings.shape, (36, 768))
        assert_array_equal(embeddings.to_numpy(), 0)

    def test_no_usable_history_returns_an_empty_aligned_frame(self):
        for length in (19, 24):
            with self.subTest(length=length):
                result = self.nb.align(*raw_frames(n=length))
                self.assertTrue(result.empty)
                self.assertIn("target_direction", result.columns)
        jci, bi, inf, kur = raw_frames()
        bi.loc[0, "Period"] = "1 Januari 2030"
        self.assertTrue(self.nb.align(jci, bi, inf, kur).empty)


class TensorTests(NotebookTestCase):
    def setUp(self):
        super().setUp()
        self.nb.CFG = replace(self.nb.CFG, look_back=3, news_dim=2,
                              tech_cols=("Volume", "Close"),
                              macro_cols=("inflation_rate", "bi_rate"))
        self.frame = tensor_frame()

    def test_windows_and_all_modalities_align_to_the_window_end(self):
        before = self.frame.copy(deep=True)
        tensors, _ = self.nb.make_tensors(self.frame, train_end_idx=4)
        self.assertEqual(tensors.X_tech.shape, (4, 3, 2))
        self.assertEqual(tensors.X_macro.shape, (4, 2))
        self.assertEqual(tensors.X_news.shape, (4, 2))
        # Both features scale to these hand-computed values, including held-out extremes.
        scaled = np.array([0, 1 / 3, 2 / 3, 1, 10 / 3, -5 / 3])
        for window in range(4):
            expected = np.repeat(scaled[window:window + 3, None], 2, axis=1)
            assert_allclose(tensors.X_tech[window], expected, atol=1e-6)
        assert_allclose(tensors.X_macro, np.repeat(scaled[2:, None], 2, axis=1), atol=1e-6)
        assert_array_equal(tensors.X_news, [[12, -12], [13, -13], [14, -14], [15, -15]])
        assert_array_equal(tensors.y, [1, 0, 1, 0])
        assert_array_equal(tensors.dates, ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
        for array in (tensors.X_tech, tensors.X_macro, tensors.X_news, tensors.y):
            self.assertEqual(array.dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(array).all())
        assert_frame_equal(self.frame, before)

    def test_scalers_fit_only_the_exclusive_train_prefix(self):
        tensors, scaler = self.nb.make_tensors(self.frame, train_end_idx=4)
        assert_array_equal(scaler.data_min_, [200, 100])
        assert_array_equal(scaler.data_max_, [212, 106])
        self.assertEqual(scaler.n_samples_seen_, 4)
        # Held-out macro values must exceed [0, 1], even though its scaler isn't returned.
        self.assertTrue((tensors.X_macro[-2] > 1).all())
        self.assertTrue((tensors.X_macro[-1] < 0).all())
        changed = self.frame.copy()
        changed.loc[4:, ["Volume", "Close", "bi_rate", "inflation_rate"]] = 1e6
        later, later_scaler = self.nb.make_tensors(changed, train_end_idx=4)
        assert_array_equal(later_scaler.data_min_, scaler.data_min_)
        assert_array_equal(later_scaler.data_max_, scaler.data_max_)
        assert_array_equal(later.X_tech[:2], tensors.X_tech[:2])
        assert_array_equal(later.X_macro[:2], tensors.X_macro[:2])

    def test_feature_order_matches_config_even_for_constant_training_columns(self):
        self.frame["Volume"] = 200.0
        self.frame["inflation_rate"] = 10.0
        tensors, _ = self.nb.make_tensors(self.frame, train_end_idx=4)
        assert_array_equal(tensors.X_tech[:, :, 0], 0)
        assert_allclose(tensors.X_tech[0, :, 1], [0, 1 / 3, 2 / 3], atol=1e-6)
        assert_array_equal(tensors.X_macro[:, 0], 0)
        assert_allclose(tensors.X_macro[:, 1], [2 / 3, 1, 10 / 3, -5 / 3], atol=1e-6)

    def test_one_row_and_exact_lookback_boundaries(self):
        for look_back, rows, expected_samples in ((1, 1, 1), (3, 3, 1), (1, 6, 6)):
            with self.subTest(look_back=look_back, rows=rows):
                self.nb.CFG = replace(self.nb.CFG, look_back=look_back)
                tensors, scaler = self.nb.make_tensors(self.frame.iloc[:rows], train_end_idx=rows)
                self.assertEqual(tensors.X_tech.shape, (expected_samples, look_back, 2))
                self.assertEqual(scaler.n_samples_seen_, rows)
                self.assertEqual(tensors.y[-1], self.frame["target_direction"].iloc[rows - 1])
                self.assertTrue(np.isfinite(tensors.X_tech).all())
                self.assertTrue(np.isfinite(tensors.X_macro).all())

    def test_insufficient_history_is_rejected(self):
        with self.assertRaises(ValueError):
            self.nb.make_tensors(self.frame.iloc[:2], train_end_idx=2)

    def test_invalid_train_boundaries_and_empty_frame_are_rejected(self):
        for boundary in (-1, 0, 7):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(AssertionError, "out of range"):
                self.nb.make_tensors(self.frame, train_end_idx=boundary)
        with self.assertRaisesRegex(AssertionError, "out of range"):
            self.nb.make_tensors(self.frame.iloc[:0], train_end_idx=1)

    def test_target_columns_cannot_be_configured_as_technical_features(self):
        for forbidden in ("future_close", "return_5d", "target_direction"):
            with self.subTest(column=forbidden):
                self.nb.CFG = replace(self.nb.CFG, tech_cols=("Close", forbidden))
                with self.assertRaises(AssertionError):
                    self.nb.make_tensors(self.frame, train_end_idx=4)

    def test_future_targets_do_not_affect_features(self):
        baseline, _ = self.nb.make_tensors(self.frame, train_end_idx=4)
        changed = self.frame.copy()
        changed["future_close"] = -1e6
        changed["return_5d"] = 1e6
        changed["target_direction"] = 1 - changed["target_direction"]
        result, _ = self.nb.make_tensors(changed, train_end_idx=4)
        for field in ("X_tech", "X_macro", "X_news", "dates"):
            assert_array_equal(getattr(result, field), getattr(baseline, field))
        assert_array_equal(result.y, 1 - baseline.y)


if __name__ == "__main__":
    unittest.main()
