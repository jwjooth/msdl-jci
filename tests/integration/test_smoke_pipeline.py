"""Smoke test: tiny synthetic pipeline end-to-end in seconds."""

import numpy as np
import pandas as pd


def test_smoke_pipeline():
    from msdl_jci.evaluation.metrics import compute_classification_metrics
    from msdl_jci.evaluation.trading_simulation import simulate_trading_strategy
    from msdl_jci.evaluation.walk_forward import (
        evaluate_model_walk_forward,
        set_all_seeds,
    )
    from msdl_jci.models.fusion import PureLSTMModel
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    set_all_seeds(0)
    rng = np.random.default_rng(0)
    n = 90
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "Close": 100 + np.cumsum(rng.normal(0, 1, n)),
        "Volume": 1e6, "RSI_14": 50.0, "MACD": 0.0, "MACD_Signal": 0.0,
        "ATR_14": 2.0, "SMA_20": 100.0,
        "bi_rate": 5.0, "inflation_rate": 3.0, "usd_idr": 15000.0,
        "future_close": 0.0, "return_5d": 0.0, "target_direction": 0.0,
    })
    for i in range(4):
        df[f"emb_{i}"] = rng.normal(0, 0.1, n)
    h = 5
    df["future_close"] = df["Close"].shift(-h).values
    df["return_5d"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["target_direction"] = (df["future_close"] > df["Close"]).astype(float)
    df = df.dropna(subset=["future_close"]).reset_index(drop=True)

    b = MultiSourceDatasetBuilder()
    t, _, _ = b.create_multisource_tensors(df, train_end_idx=50)
    assert len(t.y) == len(df) - b.look_back + 1

    res = evaluate_model_walk_forward(
        lambda: PureLSTMModel(), "smoke", t, epochs=1, seed=0)
    assert len(res.predictions) == len(res.actuals) > 0
    m = compute_classification_metrics(res.actuals, res.predictions)
    assert 0.0 <= m.roc_auc <= 1.0
    fin = simulate_trading_strategy(res.predictions, t.returns_5d[-len(res.predictions):])
    assert np.isfinite(fin.total_return)
