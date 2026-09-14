"""Realistic trading documentation + plots (Phase 10).

Recomputes deterministic heuristic equity curves (no training) for
cumulative/drawdown/exposure plots; neural rows come from Phase 8 CSV tables.
Saves to reports/phase_10_trading/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_10_trading"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from msdl_jci.evaluation.trading_simulation import simulate_trading_strategy
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    (OUT / "trading_assumptions.md").write_text(
        "# Trading assumptions (frozen)\n\n"
        "- Signal at close(t) trades at close(t); 5-day stride, non-overlapping; long/flat only.\n"
        "- Costs: fee 0.0015/side + slippage 0.0005/side (round-trip 0.004 on position open+close).\n"
        "- Cash earns 5% annualized risk-free. Benchmark: buy-and-hold on same stride dates.\n"
        "- Gross = pre-cost compounding; net = post-cost. Exposure = % steps long; "
        "turnover = trades/steps. Cost drag = gross − net (pp).\n"
        "- Verdict rule: exposure ≈100% ⇒ strategy ≈ buy-and-hold; judge on Sharpe/MDD, not return.\n")

    b = MultiSourceDatasetBuilder()
    df = b.build_aligned_dataframe()
    lb = b.look_back
    n_win = len(df) - lb + 1
    t, _, _ = b.create_multisource_tensors(df, train_end_idx=int(n_win * 0.70) + lb - 1)
    n = len(t.y)
    te = slice(int(n * 0.80) + 5, n)
    y, r = t.y[te], t.returns_5d[te]
    rng = np.random.default_rng(42)
    probs = {
        "always-positive": np.full(len(y), 0.99),
        "momentum-5d": np.clip(0.5 + (t.X_tech[te, -1, 0] - t.X_tech[te, 0, 0]) * 2, 0.01, 0.99),
        "random": rng.uniform(0, 1, len(y)),
    }
    curves = {}
    for name, p in probs.items():
        m = simulate_trading_strategy(p, r, threshold=0.5)
        curves[name] = m
    bench = curves["always-positive"].benchmark_curve

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, m in curves.items():
        ax.plot(m.equity_curve, label=f"{name} NET")
    ax.plot(bench, "--", color="black", label="Buy-and-Hold")
    ax.set_title("Cumulative portfolio (NET, heuristics, test slice)")
    ax.set_xlabel("5-day steps")
    ax.legend()
    fig.savefig(OUT / "cumulative_returns_net.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for name, m in curves.items():
        c = m.equity_curve
        dd = (c - np.maximum.accumulate(c)) / np.maximum(np.maximum.accumulate(c), 1e-12)
        ax.plot(dd * 100, label=name)
    ax.set_title("Drawdown % (NET, heuristics)")
    ax.legend()
    fig.savefig(OUT / "drawdown_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Gross vs net + cost drag from Phase 8 full table.
    src = pd.read_csv(PROJECT_ROOT / "reports" / "phase_08_baselines" / "full_audit_comparison.csv")
    gvn = src[["Model", "Gross Return (%)", "Total Return (%)", "Total Costs (pp)",
               "Exposure (%)", "Turnover", "Sharpe Ratio", "Max Drawdown (%)"]].copy()
    gvn.columns = ["Model", "gross", "net", "cost_drag_pp", "exposure", "turnover",
                   "sharpe_net", "maxdd"]
    gvn.to_csv(OUT / "gross_vs_net_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(gvn))
    ax.bar(x - 0.2, gvn["gross"], 0.4, label="gross")
    ax.bar(x + 0.2, gvn["net"], 0.4, label="net")
    ax.set_xticks(x)
    ax.set_xticklabels(gvn["Model"], rotation=25, ha="right", fontsize=7)
    ax.set_title("Gross vs net return by model (Phase 8, test)")
    ax.legend()
    fig.savefig(OUT / "cumulative_returns_gross.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(gvn["Model"], gvn["exposure"])
    ax.set_title("Exposure % by model")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    fig.savefig(OUT / "exposure_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    worst = gvn.loc[gvn["cost_drag_pp"].idxmax()]
    (OUT / "cost_drag_report.md").write_text(
        "# Cost drag report\n\n" + gvn.to_string(index=False) +
        f"\n\n- Highest cost drag: {worst['Model']} ({worst['cost_drag_pp']:.2f} pp) — "
        "high turnover + low edge ⇒ costs dominate.\n"
        "- Proposed net (3.50) vs gross (9.03): 5.53 pp drag wipes out >60% of gross edge.\n"
        "- momentum-5d: 1.55 pp drag, Sharpe 1.308 net — cheapest edge per trade.\n"
        "- Any model with exposure ≈100% (PureLSTM/LSTM+Macro/Static here) is buy-and-hold "
        "with extra costs; its 'return' is benchmark return minus drag.\n")
    print(gvn.to_string(index=False))


if __name__ == "__main__":
    main()
