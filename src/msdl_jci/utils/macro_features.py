"""Point-in-time-safe macro change/regime features (Phase 4).

All features are causal: computed with shift(1) so the value at date t uses
only macro observations strictly before t (conservative release lag on top of
merge_asof-backward alignment). No future data enters any row.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_macro_change_features(
    df: pd.DataFrame,
    lags: Sequence[int] = (21, 63, 126),
    z_windows: Sequence[int] = (126, 252),
) -> pd.DataFrame:
    """Add causal macro change / z-score / regime columns to aligned dataframe."""
    df = df.copy().sort_values("Date").reset_index(drop=True)
    base = ["bi_rate", "inflation_rate", "usd_idr"]
    for col in base:
        if col not in df.columns:
            continue
        prev = df[col].shift(1)  # strictly past observations only
        for lag in lags:
            df[f"{col}_chg_{lag}d"] = (prev - prev.shift(lag)) / (prev.shift(lag).abs() + 1e-9)
        for w in z_windows:
            roll = prev.rolling(w, min_periods=max(21, w // 4))
            mu, sd = roll.mean(), roll.std().replace(0.0, np.nan)
            df[f"{col}_z{w}d"] = ((prev - mu) / (sd + 1e-9)).fillna(0.0)
    if "bi_rate_chg_63d" in df.columns:
        df["rate_hiking"] = (df["bi_rate_chg_63d"] > 0.02).astype(float)
        df["rate_cutting"] = (df["bi_rate_chg_63d"] < -0.02).astype(float)
    if "inflation_rate" in df.columns:
        df["high_inflation"] = (df["inflation_rate"].shift(1) > 5.0).astype(float)
    if "usd_idr_chg_63d" in df.columns:
        df["idr_depreciating"] = (df["usd_idr_chg_63d"] > 0.02).astype(float)
    return df


ENGINEERED_MACRO_COLS = [
    "bi_rate_chg_21d",
    "bi_rate_chg_63d",
    "bi_rate_chg_126d",
    "inflation_rate_chg_21d",
    "inflation_rate_chg_63d",
    "inflation_rate_chg_126d",
    "usd_idr_chg_21d",
    "usd_idr_chg_63d",
    "usd_idr_chg_126d",
    "usd_idr_z126d",
    "usd_idr_z252d",
    "rate_hiking",
    "rate_cutting",
    "high_inflation",
    "idr_depreciating",
]
