# Macro feature proposals (causal; see utils/macro_features.py)

- Change features: `{bi_rate,inflation_rate,usd_idr}_chg_{21,63,126}d` on shift(1) values.
- Z-scores: `usd_idr_z{126,252}d` vs rolling past-only mean/std.
- Regimes: `rate_hiking/cutting`, `high_inflation`, `idr_depreciating`.
- Capacity: shrink macro MLP (e.g. 3->16->8), dropout 0.2, early stopping;
  stale levels alone cannot justify gating dominance.
