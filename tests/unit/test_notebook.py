"""Unit tests for the implementation in development.ipynb.

Uses unittest from the standard library plus the notebook's declared dependencies.
From the repository root: python -m unittest discover -s tests/unit -p 'test_notebook.py'
The loader executes only definition cells; it never executes the run or plot cells,
reads the user's .env, or opens the user's database. SQLite fixtures are temporary.
"""

import io
import json
import os
import sqlite3
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import call, patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "development.ipynb"
# Independent schema oracle: do not construct fixtures from ENTITY_TABLES.
SCHEMA = {
    "jci_historical": ("Date", "Close", "High", "Low", "Open", "Volume"),
    "bi_rate": ("Period", "BI-7Day-RR"),
    "inflation_data": ("Periode", "Data Inflasi"),
    "kurs_usdidr": ("Date", "Close", "High", "Low", "Open"),
    "cnbc_ihsg_articles": ("title", "url", "publish_date", "content", "scraped_at"),
    "detik_ihsg_articles": ("title", "url", "snippet", "published", "scraped_at"),
    "kontan_ihsg_articles": ("title", "url", "snippet", "published", "category", "scraped_at"),
}


def load_definitions(root, env=None, cwd=None):
    """Load actual cells by stable notebook IDs in a fresh, isolated namespace."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
             if cell["cell_type"] == "code"}
    module = ModuleType("_notebook_under_test")
    with (
        patch.dict(os.environ, {"MPLBACKEND": "Agg", **(env or {})}, clear=True),
        patch.dict(sys.modules, {module.__name__: module}),
        patch("dotenv.load_dotenv", return_value=False),
        patch.object(Path, "cwd", return_value=cwd or root),
        redirect_stdout(io.StringIO()),
    ):
        for cell_id in ("imports", "config", "entity", "utils"):
            # The only non-Python line is an IPython plotting directive.
            source = "\n".join(line for line in cells[cell_id].splitlines()
                               if line.strip() != "%matplotlib inline")
            exec(compile(source, f"{NOTEBOOK}#{cell_id}", "exec"), module.__dict__)
    return module


def market_frames(n=60):
    """Small chronological data with macro values available before warmup ends."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 + np.arange(n)
    jci = pd.DataFrame({"Date": dates, "Close": close, "Open": close - 1,
                        "High": close + 2, "Low": close - 2, "Volume": 1000.0})
    bi = pd.DataFrame({"Period": ["1 Januari 2024"], "BI-7Day-RR": [" 5.5 % "]})
    inf = pd.DataFrame({"Periode": ["Januari 2024"], "Data Inflasi": [" 2.5 % "]})
    kur = pd.DataFrame({"Date": dates, "Close": ["15,000"] * n})
    return jci, bi, inf, kur


def tensor_frame(cfg, n=6):
    """Distinct feature scales, labels and embeddings expose ordering mistakes."""
    steps = np.arange(n, dtype=float)
    values = {col: 10 * (i + 1) + (i + 1) * steps
              for i, col in enumerate(cfg.tech_cols + cfg.macro_cols)}
    values.update({"Date": pd.date_range("2024-01-01", periods=n, freq="B"),
                   "target_direction": steps % 2, "future_close": steps + 999,
                   "return_5d": steps / 100, "emb_0": steps + 10, "emb_1": -steps})
    return pd.DataFrame(values)


class NotebookTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").touch()
        self.nb = load_definitions(self.root)


class ConfigTests(NotebookTestCase):
    def test_defaults_and_immutable_config(self):
        cfg = self.nb.CFG
        self.assertEqual((cfg.look_back, cfg.horizon, cfg.seed, cfg.news_dim), (28, 5, 42, 768))
        self.assertEqual(cfg.tech_cols,
                         ("Close", "Volume", "RSI_14", "MACD", "MACD_Signal", "ATR_14", "SMA_20"))
        self.assertEqual(cfg.macro_cols, ("bi_rate", "inflation_rate", "usd_idr"))
        self.assertEqual(self.nb.DB_PATH, self.root / "database/main_database.db")
        self.assertFalse(self.nb.USE_DB)
        with self.assertRaises(FrozenInstanceError):
            cfg.look_back = 1

    def test_environment_overrides_and_nested_root_discovery(self):
        nested = self.root / "notebooks" / "nested"
        nested.mkdir(parents=True)
        nb = load_definitions(self.root, {"DATABASE_PATH": "fixtures/custom.db",
                              "ML_LOOK_BACK": "7", "ML_PREDICTION_HORIZON": "2",
                              "ML_RANDOM_STATE": "19"}, cwd=nested)
        self.assertEqual(nb.NB_ROOT, self.root)
        self.assertEqual(nb.DB_PATH, self.root / "fixtures/custom.db")
        self.assertEqual((nb.CFG.look_back, nb.CFG.horizon, nb.CFG.seed), (7, 2, 19))

    def test_absolute_database_path_and_existing_source(self):
        db = self.root / "existing.db"
        with sqlite3.connect(db):
            pass
        nb = load_definitions(self.root, {"DATABASE_PATH": str(db)})
        self.assertEqual(nb.DB_PATH, db)
        self.assertTrue(nb.USE_DB)

    def test_invalid_integer_environment_is_rejected(self):
        for name in ("ML_LOOK_BACK", "ML_PREDICTION_HORIZON", "ML_RANDOM_STATE"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_definitions(self.root, {name: "not-an-integer"})


class SQLiteTests(NotebookTestCase):
    def setUp(self):
        super().setUp()
        self.nb.DB_PATH = self.root / "fixture.db"

    def write_table(self, name, columns, rows=1):
        frame = pd.DataFrame({col: [f"{col}-{i}" for i in range(rows)] for col in columns})
        with sqlite3.connect(self.nb.DB_PATH) as con:
            frame.to_sql(name, con, index=False, if_exists="replace")
        return frame

    def test_all_entity_contracts_read_rows_and_preserve_extra_columns(self):
        self.assertEqual(self.nb.ENTITY_TABLES, SCHEMA)
        for name, columns in SCHEMA.items():
            with self.subTest(table=name):
                expected = self.write_table(name, (*columns, "extra"), rows=2)
                assert_frame_equal(self.nb.read_table(name), expected)

    def test_every_required_column_is_enforced(self):
        for name, columns in SCHEMA.items():
            for missing in columns:
                with self.subTest(table=name, missing=missing):
                    self.write_table(name, [col for col in columns if col != missing])
                    with self.assertRaises(ValueError) as caught:
                        self.nb.read_table(name)
                    self.assertIn(f"Table [{name}] missing columns", str(caught.exception))
                    self.assertIn(repr(missing), str(caught.exception))

    def test_empty_table_with_valid_schema_is_accepted(self):
        self.write_table("bi_rate", SCHEMA["bi_rate"], rows=0)
        result = self.nb.read_table("bi_rate")
        self.assertTrue(result.empty)
        self.assertEqual(tuple(result.columns), SCHEMA["bi_rate"])

    def test_missing_table_reports_database_error(self):
        with self.assertRaises(pd.errors.DatabaseError):
            self.nb.read_table("jci_historical")

    def test_database_loader_reads_four_market_tables_in_order(self):
        names = ("jci_historical", "bi_rate", "inflation_data", "kurs_usdidr")
        frames = [pd.DataFrame({"source": [name]}) for name in names]
        self.nb.USE_DB = True
        with patch.object(self.nb, "read_table", side_effect=frames) as read:
            actual = self.nb.load_frames()
        self.assertEqual(read.call_args_list, [call(name) for name in names])
        for result, expected in zip(actual, frames, strict=True):
            self.assertIs(result, expected)

    def test_invalid_database_does_not_silently_fall_back(self):
        self.nb.USE_DB = True
        with patch.object(self.nb, "read_table", side_effect=ValueError("schema drift")):
            with self.assertRaisesRegex(ValueError, "schema drift"):
                self.nb.load_frames()


class SyntheticDataTests(NotebookTestCase):
    def test_fallback_is_reproducible_and_never_reads_sqlite(self):
        with patch.object(self.nb, "read_table", side_effect=AssertionError("unexpected DB read")):
            first = self.nb.load_frames()
            second = self.nb.load_frames()
        for actual, expected in zip(first, second, strict=True):
            assert_frame_equal(actual, expected)
        jci, bi, inf, kur = first
        self.assertEqual(tuple(map(len, first)), (500, 17, 17, 500))
        self.assertEqual(tuple(jci.columns), ("Date", "Open", "High", "Low", "Close", "Volume"))
        assert_series_equal(jci["Date"], kur["Date"])
        self.assertTrue((jci["Date"].dt.dayofweek < 5).all())
        self.assertTrue((jci["Low"] < jci["Close"]).all())
        self.assertTrue((jci["High"] > jci["Close"]).all())
        self.assertTrue(self.nb.parse_id_date(bi["Period"]).notna().all())
        self.assertTrue(self.nb.parse_id_date(inf["Periode"], day_first=False).notna().all())

    def test_seed_changes_both_market_series(self):
        first = self.nb.load_frames()
        self.nb.CFG = replace(self.nb.CFG, seed=43)
        second = self.nb.load_frames()
        for index in (0, 3):
            self.assertFalse(first[index]["Close"].equals(second[index]["Close"]))
        for index in (1, 2):
            assert_frame_equal(first[index], second[index])


class DateParsingTests(NotebookTestCase):
    def test_all_indonesian_month_names_and_aliases(self):
        aliases = ["Januari jan", "Februari feb", "Maret mar", "April apr", "Mei may",
                   "Juni jun", "Juli jul", "Agustus agu aug", "September sep",
                   "Oktober okt oct", "November nov", "Desember des dec"]
        for month, names in enumerate(aliases, start=1):
            for name in names.split():
                with self.subTest(month=month, alias=name):
                    actual = self.nb.parse_id_date(pd.Series([f"  15 {name.upper()} 2024  "]))
                    self.assertEqual(actual.iloc[0], pd.Timestamp(2024, month, 15))

    def test_month_only_iso_and_leap_day_dates(self):
        actual = self.nb.parse_id_date(
            pd.Series(["Februari 2024", "2024-03-25", "29 Februari 2024"]), day_first=False)
        expected = pd.Series(pd.to_datetime(["2024-02-01", "2024-03-25", "2024-02-29"]))
        assert_series_equal(actual, expected)

    def test_ambiguous_numeric_dates_respect_day_first(self):
        values = pd.Series(["03/04/2024"])
        self.assertEqual(self.nb.parse_id_date(values).iloc[0], pd.Timestamp("2024-04-03"))
        self.assertEqual(self.nb.parse_id_date(values, False).iloc[0], pd.Timestamp("2024-03-04"))

    def test_invalid_dates_are_coerced_and_index_is_preserved(self):
        values = pd.Series([None, "", "not a date", "31 Februari 2024"], index=[4, 8, 12, 16])
        actual = self.nb.parse_id_date(values)
        self.assertTrue(actual.isna().all())
        self.assertTrue(actual.index.equals(values.index))


class IndicatorTests(NotebookTestCase):
    def test_warmup_and_constant_price_indicators(self):
        frame = market_frames(25)[0]
        frame[["Close", "High", "Low"]] = [100.0, 102.0, 98.0]
        actual = self.nb.add_indicators(frame)
        self.assertTrue(actual["RSI_14"].iloc[:14].isna().all())
        self.assertTrue(actual["ATR_14"].iloc[:13].isna().all())
        self.assertTrue(actual["SMA_20"].iloc[:19].isna().all())
        assert_allclose(actual["RSI_14"].iloc[14:], 0)
        assert_allclose(actual["ATR_14"].iloc[13:], 4)
        assert_allclose(actual["SMA_20"].iloc[19:], 100)
        assert_allclose(actual[["MACD", "MACD_Signal"]], 0)

    def test_rising_and_falling_prices_have_expected_rsi_and_sma(self):
        for step, rsi in ((1, 100), (-1, 0)):
            with self.subTest(step=step):
                frame = market_frames(25)[0]
                frame["Close"] = 100 + step * np.arange(25)
                actual = self.nb.add_indicators(frame)
                assert_allclose(actual["RSI_14"].iloc[14:], rsi, atol=1e-6)
                self.assertAlmostEqual(actual["SMA_20"].iloc[19], 100 + step * 9.5)

    def test_macd_and_signal_first_price_change(self):
        actual = self.nb.add_indicators(market_frames(25)[0])
        expected_macd = 2 / 13 - 2 / 27
        self.assertAlmostEqual(actual["MACD"].iloc[1], expected_macd)
        self.assertAlmostEqual(actual["MACD_Signal"].iloc[1], expected_macd * 2 / 10)

    def test_atr_includes_overnight_price_gap(self):
        frame = market_frames(20)[0]
        frame[["Close", "High", "Low"]] = [100.0, 102.0, 98.0]
        frame.loc[14:, ["Close", "High", "Low"]] = [120.0, 122.0, 118.0]
        actual = self.nb.add_indicators(frame)
        self.assertAlmostEqual(actual["ATR_14"].iloc[14], 4 + (22 - 4) / 14)

    def test_future_prices_do_not_change_past_indicators(self):
        original = market_frames()[0]
        changed = original.copy(deep=True)
        changed.loc[35:, ["Close", "High", "Low"]] *= 10
        first = self.nb.add_indicators(original)
        second = self.nb.add_indicators(changed)
        assert_frame_equal(first.iloc[:35], second.iloc[:35])


class AlignmentTests(NotebookTestCase):
    def setUp(self):
        super().setUp()
        self.nb.CFG = replace(self.nb.CFG, news_dim=2)

    def test_sorted_output_warmup_tail_and_zero_news(self):
        frames = market_frames()
        actual = self.nb.align(*(frame.iloc[::-1] for frame in frames))
        self.assertEqual(len(actual), 60 - 19 - 5)
        assert_array_equal(actual["Date"], frames[0]["Date"].iloc[19:-5])
        self.assertEqual(list(actual.index), list(range(len(actual))))
        assert_allclose(actual[["emb_0", "emb_1"]], 0)
        assert_allclose(actual["bi_rate"], 5.5)
        assert_allclose(actual["inflation_rate"], 2.5)
        assert_allclose(actual["usd_idr"], 15000)
        self.assertTrue(np.isfinite(actual[list(self.nb.CFG.tech_cols)]).all().all())

    def test_default_news_dimension_is_768(self):
        self.nb.CFG = self.nb.Config()
        actual = self.nb.align(*market_frames(30))
        news = actual.filter(regex=r"^emb_")
        self.assertEqual(list(news.columns), [f"emb_{i}" for i in range(768)])
        assert_allclose(news, 0)

    def test_horizon_labels_use_future_trading_rows_including_ties(self):
        self.nb.CFG = replace(self.nb.CFG, horizon=2)
        frames = market_frames(35)
        # Dates 19..23: up, down, tie when compared to two trading rows later.
        frames[0].loc[19:23, "Close"] = [100, 110, 120, 90, 120]
        actual = self.nb.align(*frames)
        assert_allclose(actual["future_close"].iloc[:3], [120, 90, 120])
        assert_allclose(actual["target_direction"].iloc[:3], [1, 0, 0])
        assert_allclose(actual["return_5d"].iloc[:3], [0.2, -20 / 110, 0])
        self.assertEqual(actual["Date"].iloc[-1], frames[0]["Date"].iloc[-3])

    def test_macro_release_is_used_on_release_date_but_never_earlier(self):
        jci, _, _, kur = market_frames()
        bi = pd.DataFrame({"Period": ["1 Februari 2024", "1 Januari 2024"],
                           "BI-7Day-RR": ["9 %", "5 %"]})
        inf = pd.DataFrame({"Periode": ["Februari 2024", "Januari 2024"],
                            "Data Inflasi": ["7 %", "2 %"]})
        actual = self.nb.align(jci, bi, inf, kur).set_index("Date")
        assert_allclose(actual.loc["2024-01-31", ["bi_rate", "inflation_rate"]].astype(float),
                        [5, 2])
        assert_allclose(actual.loc["2024-02-01", ["bi_rate", "inflation_rate"]].astype(float),
                        [9, 7])

    def test_each_macro_source_drops_leading_gaps_instead_of_backfilling(self):
        for source in ("bi", "inf", "kur"):
            with self.subTest(source=source):
                jci, bi, inf, kur = market_frames()
                if source == "bi":
                    bi["Period"] = "1 Februari 2024"
                elif source == "inf":
                    inf["Periode"] = "Februari 2024"
                else:
                    kur = kur.iloc[23:]
                actual = self.nb.align(jci, bi, inf, kur)
                self.assertEqual(actual["Date"].iloc[0], pd.Timestamp("2024-02-01"))

    def test_currency_gaps_and_invalid_values_use_last_known_value(self):
        jci, bi, inf, _ = market_frames()
        kur = pd.DataFrame({"Date": jci["Date"].iloc[[0, 21, 23]],
                            "Close": ["15,000", "invalid", "16,000"]})
        actual = self.nb.align(jci, bi, inf, kur).set_index("Date")
        self.assertEqual(actual.loc[jci["Date"].iloc[22], "usd_idr"], 15000)
        self.assertEqual(actual.loc[jci["Date"].iloc[23], "usd_idr"], 16000)

    def test_bad_dates_and_missing_close_are_discarded(self):
        frames = market_frames()
        expected = self.nb.align(*frames)
        dirty = []
        for frame in frames:
            bad = frame.iloc[[0]].copy()
            date_col = next(col for col in ("Date", "Period", "Periode") if col in bad)
            bad[date_col] = "invalid-date"
            dirty.append(pd.concat([frame, bad], ignore_index=True))
        missing = frames[0].iloc[[0]].copy()
        missing["Close"] = np.nan
        dirty[0] = pd.concat([dirty[0], missing], ignore_index=True)
        assert_frame_equal(self.nb.align(*dirty), expected)

    def test_alignment_does_not_mutate_any_input(self):
        frames = market_frames()
        copies = [frame.copy(deep=True) for frame in frames]
        self.nb.align(*frames)
        for actual, expected in zip(frames, copies, strict=True):
            assert_frame_equal(actual, expected)

    def test_no_macro_observations_before_market_end_yields_empty_frame(self):
        jci, bi, inf, kur = market_frames()
        bi["Period"] = "1 Januari 2030"
        actual = self.nb.align(jci, bi, inf, kur)
        self.assertTrue(actual.empty)
        self.assertIn("future_close", actual.columns)

    def test_insufficient_indicator_history_yields_empty_frame(self):
        self.assertTrue(self.nb.align(*market_frames(19)).empty)


class TensorTests(NotebookTestCase):
    def setUp(self):
        super().setUp()
        self.nb.CFG = replace(self.nb.CFG, look_back=3, news_dim=2)
        self.frame = tensor_frame(self.nb.CFG)

    def test_window_values_and_endpoint_modalities_labels_and_dates(self):
        tensors, _ = self.nb.make_tensors(self.frame, train_end_idx=3)
        self.assertEqual(tensors.X_tech.shape, (4, 3, 7))
        self.assertEqual(tensors.X_macro.shape, (4, 3))
        self.assertEqual(tensors.X_news.shape, (4, 2))
        # Training rows are affine sequences: all features normalize to 0, .5, 1.
        for window in range(4):
            expected = np.repeat(((np.arange(3) + window) / 2)[:, None], 7, axis=1)
            assert_allclose(tensors.X_tech[window], expected)
        assert_allclose(tensors.X_macro, np.repeat(np.array([[1], [1.5], [2], [2.5]]), 3, axis=1))
        assert_array_equal(tensors.X_news, [[12, -2], [13, -3], [14, -4], [15, -5]])
        assert_array_equal(tensors.y, [0, 1, 0, 1])
        assert_array_equal(tensors.dates, ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
        for array in (tensors.X_tech, tensors.X_macro, tensors.X_news, tensors.y):
            self.assertEqual(array.dtype, np.float32)

    def test_scalers_ignore_holdout_extremes_without_clipping(self):
        technical = list(self.nb.CFG.tech_cols)
        macro = list(self.nb.CFG.macro_cols)
        baseline, _ = self.nb.make_tensors(self.frame, 3)
        self.frame.loc[3, technical + macro] = 10000
        self.frame.loc[4, technical + macro] = -10000
        actual, scaler = self.nb.make_tensors(self.frame, 3)
        assert_allclose(scaler.data_min_, self.frame[technical].iloc[0])
        assert_allclose(scaler.data_max_, self.frame[technical].iloc[2])
        self.assertEqual(scaler.n_samples_seen_, 3)
        assert_allclose(actual.X_tech[0], baseline.X_tech[0])
        assert_allclose(actual.X_macro[0], baseline.X_macro[0])
        self.assertTrue((actual.X_tech[1, -1] > 1).all())
        self.assertTrue((actual.X_tech[2, -1] < 0).all())
        self.assertTrue((actual.X_macro[1] > 1).all())
        self.assertTrue((actual.X_macro[2] < 0).all())

    def test_constant_training_features_remain_finite(self):
        self.frame[list(self.nb.CFG.tech_cols + self.nb.CFG.macro_cols)] = 7.0
        actual, _ = self.nb.make_tensors(self.frame, 3)
        assert_allclose(actual.X_tech, 0)
        assert_allclose(actual.X_macro, 0)

    def test_exact_lookback_length_produces_one_sample(self):
        actual, _ = self.nb.make_tensors(self.frame.iloc[:3], 3)
        self.assertEqual(actual.X_tech.shape, (1, 3, 7))
        assert_array_equal(actual.y, [0])
        assert_array_equal(actual.dates, ["2024-01-03"])

    def test_lookback_one_includes_every_row(self):
        self.nb.CFG = replace(self.nb.CFG, look_back=1)
        actual, _ = self.nb.make_tensors(self.frame, 3)
        self.assertEqual(actual.X_tech.shape, (6, 1, 7))
        assert_array_equal(actual.y, self.frame["target_direction"])
        self.assertEqual(actual.dates[0], "2024-01-01")

    def test_insufficient_window_history_is_rejected(self):
        with self.assertRaises(ValueError):
            self.nb.make_tensors(self.frame.iloc[:2], 2)

    def test_training_prefix_boundaries_are_accepted(self):
        for end in (1, len(self.frame)):
            with self.subTest(end=end):
                actual, scaler = self.nb.make_tensors(self.frame, end)
                self.assertEqual(scaler.n_samples_seen_, end)
                self.assertTrue(np.isfinite(actual.X_tech).all())
                self.assertTrue(np.isfinite(actual.X_macro).all())

    def test_invalid_training_prefix_is_rejected(self):
        for end in (-1, 0, len(self.frame) + 1):
            with self.subTest(end=end), self.assertRaisesRegex(AssertionError, "out of range"):
                self.nb.make_tensors(self.frame, end)
        with self.assertRaisesRegex(AssertionError, "out of range"):
            self.nb.make_tensors(self.frame.iloc[:0], 1)

    def test_each_target_derived_technical_feature_is_rejected(self):
        for column in ("future_close", "return_5d", "target_direction"):
            with self.subTest(column=column):
                self.nb.CFG = replace(self.nb.CFG, tech_cols=("Close", column))
                with self.assertRaises(AssertionError):
                    self.nb.make_tensors(self.frame, 3)

    def test_feature_order_follows_config_and_input_is_unchanged(self):
        self.nb.CFG = replace(self.nb.CFG, tech_cols=("Volume", "Close"),
                              macro_cols=("usd_idr", "bi_rate"))
        self.frame.loc[1, "Volume"] = 21  # Different normalized value from Close.
        self.frame.loc[2, "usd_idr"] = 135
        before = self.frame.copy(deep=True)
        actual, scaler = self.nb.make_tensors(self.frame, 3)
        assert_allclose(scaler.data_min_, [20, 10])
        assert_allclose(actual.X_tech[0, 1], [0.25, 0.5])
        # Row 3 USD/IDR is 130; the training range is 100..135.
        assert_allclose(actual.X_macro[1], [30 / 35, 1.5])
        assert_frame_equal(self.frame, before)


if __name__ == "__main__":
    unittest.main()
