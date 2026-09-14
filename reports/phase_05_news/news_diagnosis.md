# News diagnosis

- Coverage: 1908 rows, 135 zero-vector days (7.1%).
- Probes (test): linear: AUC=0.5048 MCC=0.0344; mlp: AUC=0.5376 MCC=0.068; aggregated(norm5d+missing): AUC=0.499 MCC=-0.0165.
- Missingness: exact-date left-merge, zero-vector fill, no forward-fill, no missing flag in model input (flag only in aggregated probe).
- Leakage re-check: merge is on exact `Date`; embedding for date t comes from that date's row only. Residual risk: whether vendors' daily files include post-close articles — unverifiable from here; treat news as same-day public info, never intraday.
- Verdict: news carries weak/no standalone signal on this split; high-dim sparse input justifies the projection bottleneck, not a larger news branch.
