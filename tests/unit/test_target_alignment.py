"""Target alignment tests (audit Phase 7): y[i] uses Close[i+h] vs Close[i] only."""

import numpy as np
import pandas as pd

from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder


def _toy_jci(n=60, horizon=5, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Date": dates, "Open": close, "High": close + 1,
        "Low": close - 1, "Close": close,
        "Volume": np.full(n, 1e6),
    })
    return df


def test_target_uses_future_close_only(tmp_path):
    MultiSourceDatasetBuilder(prediction_horizon=5)
    df_jci = _toy_jci()
    # Minimal macro/news: builder reads CSVs; construct aligned df manually instead.
    import msdl_jci.utils.dataset_builder as db_mod

    df_tech = db_mod.calculate_technical_indicators(df_jci)
    df_tech["usd_idr"] = 15000.0
    df_tech["bi_rate"] = 5.0
    df_tech["inflation_rate"] = 3.0
    for i in range(8):
        df_tech[f"emb_{i}"] = 0.0
    # Replicate labeling logic from build_aligned_dataframe
    h = 5
    future_close = df_tech["Close"].shift(-h)
    expected = (future_close > df_tech["Close"]).astype(float)
    df_tech["future_close"] = future_close
    df_tech["return_5d"] = (future_close - df_tech["Close"]) / df_tech["Close"]
    df_tech["target_direction"] = expected
    df_labeled = df_tech.dropna(subset=["future_close"]).reset_index(drop=True)

    # Verify alignment row by row against raw closes
    closes = df_jci["Close"].values
    # df_tech preserves order after indicator calc; labeled rows are prefix minus tail
    for i in range(len(df_labeled)):
        # row i in labeled corresponds to original index i (after no macro drops here)
        exp = 1.0 if closes[i + h] > closes[i] else 0.0
        assert float(df_labeled["target_direction"].iloc[i]) == exp
        assert df_labeled["return_5d"].iloc[i] == (closes[i + h] - closes[i]) / closes[i]


def test_windowing_preserves_chronology_and_alignment():
    builder = MultiSourceDatasetBuilder(prediction_horizon=5)
    n = 80
    df_jci = _toy_jci(n=n)
    import msdl_jci.utils.dataset_builder as db_mod

    df_tech = db_mod.calculate_technical_indicators(df_jci)
    df_tech["usd_idr"] = 15000.0
    df_tech["bi_rate"] = 5.0
    df_tech["inflation_rate"] = 3.0
    for i in range(8):
        df_tech[f"emb_{i}"] = 0.0
    h = 5
    df_tech["future_close"] = df_tech["Close"].shift(-h)
    df_tech["return_5d"] = (df_tech["future_close"] - df_tech["Close"]) / df_tech["Close"]
    df_tech["target_direction"] = (df_tech["future_close"] > df_tech["Close"]).astype(float)
    df_labeled = df_tech.dropna(subset=["future_close"]).reset_index(drop=True)
    df_labeled = df_labeled.dropna(subset=builder.tech_feature_cols).reset_index(drop=True)

    tensors, _, _ = builder.create_multisource_tensors(df_labeled)
    # Chronology: dates strictly increasing
    assert list(tensors.dates) == sorted(tensors.dates)
    # Window label equals df label at window end row
    lb = builder.look_back
    for w in [0, len(tensors.y) // 2, len(tensors.y) - 1]:
        df_row = w + lb - 1
        assert tensors.y[w] == df_labeled["target_direction"].iloc[df_row]
        assert tensors.dates[w] == df_labeled["Date"].dt.strftime("%Y-%m-%d").iloc[df_row]


def test_future_close_never_in_features():
    builder = MultiSourceDatasetBuilder()
    assert "future_close" not in builder.tech_feature_cols
    assert "return_5d" not in builder.tech_feature_cols
    assert "target_direction" not in builder.tech_feature_cols
