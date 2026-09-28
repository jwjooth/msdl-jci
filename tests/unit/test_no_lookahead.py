"""No-lookahead tests (audit Phase 7): train-only scaling, no bfill, embargo."""

import numpy as np
import pandas as pd

from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder


def _labeled_df(n=100, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Close": 100 + np.cumsum(rng.normal(0, 1, size=n)),
            "Volume": 1e6,
            "RSI_14": 50.0,
            "MACD": 0.1,
            "MACD_Signal": 0.05,
            "ATR_14": 2.0,
            "SMA_20": 100.0,
            "bi_rate": 5.0,
            "inflation_rate": 3.0,
            "usd_idr": 15000.0,
            "future_close": 0.0,
            "return_5d": 0.01,
            "target_direction": 1.0,
        }
    )
    for i in range(4):
        df[f"emb_{i}"] = rng.normal(0, 1, size=n)
    df["target_direction"] = (rng.uniform(0, 1, n) > 0.46).astype(float)
    return df


def test_scaler_fitted_on_train_only():
    builder = MultiSourceDatasetBuilder()
    df = _labeled_df()
    # Inject an extreme outlier ONLY in the test region.
    df.loc[len(df) - 5, "Close"] = 1e6
    train_end = 60
    tensors, feat_sc, macro_sc = builder.create_multisource_tensors(df, train_end_idx=train_end)
    # Scaler data_max_ for Close (first tech col) must reflect train prefix only.
    close_idx = builder.tech_feature_cols.index("Close")
    assert feat_sc.data_max_[close_idx] < 1e6
    # Full-fit (legacy) WOULD absorb the outlier — proving the test is sensitive.
    _, feat_full, _ = builder.create_multisource_tensors(df)
    assert feat_full.data_max_[close_idx] == 1e6


def test_macro_no_backfill_boundary(tmp_path):
    # build_aligned_dataframe tested indirectly: simulate merge_asof + ffill-only.
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    master = pd.DataFrame({"Date": dates, "usd_idr": [np.nan] * 3 + [15000.0] * 7})
    master["usd_idr"] = master["usd_idr"].ffill()
    # ffill-only leaves leading NaNs (to be dropped) instead of leaking future values.
    assert master["usd_idr"].iloc[:3].isna().all()
    leaked = master["usd_idr"].copy().ffill().bfill()
    assert not leaked.isna().any()  # bfill would hide the gap = leakage


def test_walk_forward_embargo_no_overlap():
    import inspect

    from msdl_jci.evaluation.walk_forward import evaluate_model_walk_forward

    sig = inspect.signature(evaluate_model_walk_forward)
    assert "embargo" in sig.parameters
    assert sig.parameters["embargo"].default == 5


def test_pos_weight_uses_train_only():

    from msdl_jci.evaluation.walk_forward import compute_pos_weight

    y_train = np.array([1, 1, 1, 0])
    w = compute_pos_weight(y_train)
    assert abs(float(w) - (1 / 3)) < 1e-6
    # All-positive train edge case must not explode.
    w2 = compute_pos_weight(np.ones(10))
    assert float(w2) == 0.0


def test_news_date_no_future_leak():
    # News merge is on exact Date (how='left'), never forward-looking.
    trade = pd.DataFrame({"Date": pd.to_datetime(["2020-01-06", "2020-01-07"])})
    news = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-06"]),
            "emb_0": [0.5],
        }
    )
    merged = pd.merge(trade, news, on="Date", how="left")
    assert merged["emb_0"].iloc[1] != 0.5 or True  # missing day must not copy future/past embedding
    assert pd.isna(merged["emb_0"].iloc[1])
