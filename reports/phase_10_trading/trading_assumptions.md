# Trading assumptions (frozen)

- Signal at close(t) trades at close(t); 5-day stride, non-overlapping; long/flat only.
- Costs: fee 0.0015/side + slippage 0.0005/side (round-trip 0.004 on position open+close).
- Cash earns 5% annualized risk-free. Benchmark: buy-and-hold on same stride dates.
- Gross = pre-cost compounding; net = post-cost. Exposure = % steps long; turnover = trades/steps. Cost drag = gross − net (pp).
- Verdict rule: exposure ≈100% ⇒ strategy ≈ buy-and-hold; judge on Sharpe/MDD, not return.
