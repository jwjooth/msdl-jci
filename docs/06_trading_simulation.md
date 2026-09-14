# 06 — Trading Simulation

- **Execution.** Signal at close(t) → position for next 5-day stride; long/flat only.
- **Costs.** Fee 0.0015/side + slippage 0.0005/side (round-trip 0.004 on
  open+close); cash earns 5% annualized; benchmark = buy-and-hold, same dates.
- **Reported.** Gross vs net, cost drag (pp), Sharpe/Sortino (net), MDD, win rate,
  trades, turnover, exposure.
- **Findings.** Proposed net 3.50 vs gross 9.03 (5.53 pp drag, >60% of edge) vs
  benchmark 25.32. ~100%-exposure rows ≈ buy-and-hold minus costs. Momentum-5d:
  1.55 pp drag, Sharpe 1.31 — cheapest edge per trade, still a heuristic needing
  validation. See `reports/phase_10_trading/`.
