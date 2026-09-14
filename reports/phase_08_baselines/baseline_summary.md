# Baseline summary (Phase 8 — single split, leak-free, 5-epoch neural, test slice)

- Neural nets at 5 epochs mostly collapse to always-positive (PureLSTM, LSTM+Macro,
  Static: 100% exposure, MCC=0, identical to always-positive net 24.82).
  At 40 epochs (reported run) they diverge slightly but stay weak (best AUC 0.567).
- Only LSTM+News shows spread (pos 24–46%, MCC up to 0.14, AUC 0.54).
- Proposed: AUC 0.5288, MCC 0.0155, p_std 0.0014, net 3.50 vs benchmark 25.32.
- Heuristics: momentum-5d net 28.07 (Sharpe 1.308, exposure 55%) — best risk-adjusted;
  random-forest net 27.47 but 92% exposure (≈ buy-and-hold + drag).
- Exposure verdict: any ~100%-exposure 'win' is buy-and-hold minus costs, not timing skill.
- See `baseline_classification.csv`, `baseline_trading.csv`, `full_audit_comparison.csv`.
  Net-curve plots: `reports/phase_10_trading/cumulative_returns_net.png`, `drawdown_plot.png`.
