"""Financial and economic performance simulation module for MSDL-JCI.

Implements trading strategy simulation and risk-adjusted metrics:
- Cumulative returns vs Buy-and-Hold JCI
- Annualized Sharpe Ratio
- Maximum Drawdown (MDD)
- Win Rate (%)
Matching Thesis Section 2.7.
<<<<<<< HEAD

Audit Phase 2 additions: transaction costs, slippage, gross vs net returns,
Sortino ratio, exposure %, turnover.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FinancialMetrics:
    """Financial and economic performance metrics matching Thesis Section 2.7."""

    total_return: float        # Strategy total percentage return (%) — NET of costs
    benchmark_return: float    # Buy & Hold JCI percentage return (%)
    annualized_return: float   # Compound Annual Growth Rate (%, net)
    sharpe_ratio: float        # Annualized Sharpe ratio (net)
    max_drawdown: float        # Maximum peak-to-trough decline (%)
    win_rate: float            # Percentage of winning long trades (%)
    total_trades: int          # Total trades executed
    equity_curve: np.ndarray   # Strategy cumulative value trajectory (net)
    benchmark_curve: np.ndarray  # Benchmark cumulative value trajectory
    # --- Audit Phase 2 extras ---
    gross_return: float = 0.0       # Total return before costs (%)
    total_costs: float = 0.0        # Cumulative transaction costs + slippage (% of capital drag, pp)
    sortino_ratio: float = 0.0      # Annualized Sortino ratio (downside deviation)
    exposure_pct: float = 0.0       # % of steps invested (long)
    turnover: float = 0.0           # Avg trades per step (0..1 with stride sampling)
    transaction_cost_rate: float = 0.0  # Per-side cost used
    slippage_rate: float = 0.0          # Per-side slippage used

    def to_dict(self) -> dict[str, float | int]:
        return {
            "Total Return (%)": round(self.total_return, 2),
            "Gross Return (%)": round(self.gross_return, 2),
            "Benchmark Return (%)": round(self.benchmark_return, 2),
            "Annualized Return (%)": round(self.annualized_return, 2),
            "Sharpe Ratio": round(self.sharpe_ratio, 3),
            "Sortino Ratio": round(self.sortino_ratio, 3),
            "Max Drawdown (%)": round(self.max_drawdown, 2),
            "Win Rate (%)": round(self.win_rate, 2),
            "Total Trades": int(self.total_trades),
            "Exposure (%)": round(self.exposure_pct, 2),
            "Turnover": round(self.turnover, 4),
            "Total Costs (pp)": round(self.total_costs, 2),
        }


def simulate_trading_strategy(
    y_prob: np.ndarray,
    returns_5d: np.ndarray,
    dates: np.ndarray | None = None,
    threshold: float = 0.5,
    risk_free_rate_annual: float = 0.05,
    initial_capital: float = 100.0,
    stride: int = 5,
    transaction_cost: float = 0.0015,
    slippage: float = 0.0005,
) -> FinancialMetrics:
    """Simulate realistic investment strategy over non-overlapping or stepped 5-day horizons.

    Strategy rule:
    - If P(UP) >= threshold: Take long position in JCI for the 5-day forward horizon.
    - Else: Stay in cash (0.0% return or risk-free cash return).

    Costs (audit Phase 2): entering/exiting a long position pays
    ``transaction_cost + slippage`` per side. A round-trip entered at step t
    and exited before step t+1 pays 2x one-side cost. Staying in cash is free
    (earns risk-free yield). Gross (pre-cost) and net (post-cost) returns are
    both tracked.

    Args:
        y_prob: Predicted UP probabilities [N]
        returns_5d: Forward 5-day returns [N]
        dates: Corresponding trade dates [N]
        threshold: Prediction cutoff threshold (default 0.5)
        risk_free_rate_annual: Annual risk-free rate (default 5% representing BI rate)
        initial_capital: Initial portfolio value (default 100.0)
        stride: Evaluation step stride to avoid lookahead overlapping return bias (default 5 days)
        transaction_cost: Per-side broker cost fraction (default 0.15% Indonesian broker-like)
        slippage: Per-side slippage fraction (default 0.05%)

    Returns:
        FinancialMetrics instance
    """
    y_prob = np.asarray(y_prob)
    returns_5d = np.asarray(returns_5d)
    n = len(y_prob)

    # Subsample non-overlapping 5-day trading steps
    indices = list(range(0, n, stride))
    if len(indices) < 2:
        indices = list(range(n))

    strat_val = initial_capital
    gross_val = initial_capital
    bench_val = initial_capital
    strat_history = [strat_val]
    bench_history = [bench_val]

    rf_step = (1.0 + risk_free_rate_annual) ** (stride / 252.0) - 1.0
    one_side_cost = transaction_cost + slippage

    step_returns_net = []
    step_returns_gross = []
    trade_outcomes = []
    in_position = False
    n_long_steps = 0

    for idx in indices:
        p = y_prob[idx]
        ret = returns_5d[idx]

        # Benchmark always holds JCI
        bench_val = bench_val * (1.0 + ret)
        bench_history.append(bench_val)

        # Strategy decision
        want_long = bool(p >= threshold)
        if want_long:
            n_long_steps += 1
            gross_ret = ret
            # Pay entry cost if newly entering, exit cost accrues when leaving.
            # Simplified: each long step pays one-side entry+exit amortized as
            # 2x one-side only on position changes, plus 0 if holding.
            # To stay conservative and simple: charge round-trip cost per
            # newly-opened position, and exit cost when closing.
            cost = 0.0
            if not in_position:
                cost += one_side_cost  # entry
            # If next decision will be cash, exit cost applies at close; we
            # approximate by charging half spread now is wrong — instead track:
            # charge entry now; exit charged when position closes (next cash step).
            gross_val = gross_val * (1.0 + gross_ret)
            net_ret = (1.0 + gross_ret) * (1.0 - one_side_cost if not in_position else 1.0) - 1.0
            # Peek: if we hold, no extra cost this step beyond entry already charged.
            strat_val = strat_val * (1.0 + net_ret)
            trade_outcomes.append(1 if ret > 0 else 0)
            in_position = True
            step_returns_net.append(net_ret)
            step_returns_gross.append(gross_ret)
        else:
            # Close position: pay exit cost once.
            if in_position:
                strat_val = strat_val * (1.0 - one_side_cost)
                # gross leg had no exit cost
            net_ret = rf_step
            strat_val = strat_val * (1.0 + net_ret)
            gross_val = gross_val * (1.0 + rf_step)
            in_position = False
            step_returns_net.append(net_ret)
            step_returns_gross.append(rf_step)

        strat_history.append(strat_val)

    # If still in position at end, pay final exit cost for net leg.
    if in_position:
        strat_val = strat_val * (1.0 - one_side_cost)
        strat_history[-1] = strat_val

    strat_curve = np.array(strat_history)
    bench_curve = np.array(bench_history)

    # Returns
    total_ret_net = ((strat_val - initial_capital) / initial_capital) * 100.0
    total_ret_gross = ((gross_val - initial_capital) / initial_capital) * 100.0
    bench_ret = ((bench_val - initial_capital) / initial_capital) * 100.0
    total_costs_pp = total_ret_gross - total_ret_net

    # Annualized Return (assuming 252 trading days/year)
    years = max(len(indices) * stride / 252.0, 0.1)
    cagr = ((strat_val / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    # Sharpe Ratio: Annualized (net excess returns)
    step_returns_arr = np.array(step_returns_net)
    excess_returns = step_returns_arr - rf_step
    std_ret = np.std(excess_returns)
    if std_ret > 1e-8:
        steps_per_year = 252.0 / stride
        sharpe = (np.mean(excess_returns) / std_ret) * np.sqrt(steps_per_year)
    else:
        sharpe = 0.0

    # Sortino Ratio: downside deviation only (target = rf_step)
    downside = np.minimum(0.0, step_returns_arr - rf_step)
    downside_std = np.std(downside)
    if downside_std > 1e-8:
        steps_per_year = 252.0 / stride
        sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(steps_per_year)
    else:
        sortino = 0.0

    # Maximum Drawdown (MDD) on net curve
    peaks = np.maximum.accumulate(strat_curve)
    drawdowns = (strat_curve - peaks) / np.maximum(peaks, 1e-12)
    mdd = float(np.abs(np.min(drawdowns)) * 100.0)

    # Win rate of active trades
    if trade_outcomes:
        win_rate = (sum(trade_outcomes) / len(trade_outcomes)) * 100.0
    else:
        win_rate = 0.0

    exposure = (n_long_steps / max(len(indices), 1)) * 100.0
    turnover = (len(trade_outcomes) / max(len(indices), 1))

    return FinancialMetrics(
        total_return=float(total_ret_net),
        benchmark_return=float(bench_ret),
        annualized_return=float(cagr),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(mdd),
        win_rate=float(win_rate),
        total_trades=len(trade_outcomes),
        equity_curve=strat_curve,
        benchmark_curve=bench_curve,
        gross_return=float(total_ret_gross),
        total_costs=float(total_costs_pp),
        sortino_ratio=float(sortino),
        exposure_pct=float(exposure),
        turnover=float(turnover),
        transaction_cost_rate=float(transaction_cost),
        slippage_rate=float(slippage),
||||||| c9f8f7e
=======
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class FinancialMetrics:
    """Financial and economic performance metrics matching Thesis Section 2.7."""

    total_return: float        # Strategy total percentage return (%)
    benchmark_return: float    # Buy & Hold JCI percentage return (%)
    annualized_return: float   # Compound Annual Growth Rate (%)
    sharpe_ratio: float        # Annualized Sharpe ratio
    max_drawdown: float        # Maximum peak-to-trough decline (%)
    win_rate: float            # Percentage of winning long trades (%)
    total_trades: int          # Total trades executed
    equity_curve: np.ndarray   # Strategy cumulative value trajectory
    benchmark_curve: np.ndarray  # Benchmark cumulative value trajectory

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "Total Return (%)": round(self.total_return, 2),
            "Benchmark Return (%)": round(self.benchmark_return, 2),
            "Annualized Return (%)": round(self.annualized_return, 2),
            "Sharpe Ratio": round(self.sharpe_ratio, 3),
            "Max Drawdown (%)": round(self.max_drawdown, 2),
            "Win Rate (%)": round(self.win_rate, 2),
            "Total Trades": int(self.total_trades),
        }


def simulate_trading_strategy(
    y_prob: np.ndarray,
    returns_5d: np.ndarray,
    dates: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    risk_free_rate_annual: float = 0.05,
    initial_capital: float = 100.0,
    stride: int = 5,
) -> FinancialMetrics:
    """Simulate realistic investment strategy over non-overlapping or stepped 5-day horizons.

    Strategy rule:
    - If P(UP) >= threshold: Take long position in JCI for the 5-day forward horizon.
    - Else: Stay in cash (0.0% return or risk-free cash return).

    Args:
        y_prob: Predicted UP probabilities [N]
        returns_5d: Forward 5-day returns [N]
        dates: Corresponding trade dates [N]
        threshold: Prediction cutoff threshold (default 0.5)
        risk_free_rate_annual: Annual risk-free rate (default 5% representing BI rate)
        initial_capital: Initial portfolio value (default 100.0)
        stride: Evaluation step stride to avoid lookahead overlapping return bias (default 5 days)

    Returns:
        FinancialMetrics instance
    """
    y_prob = np.asarray(y_prob)
    returns_5d = np.asarray(returns_5d)
    n = len(y_prob)

    # Subsample non-overlapping 5-day trading steps
    indices = list(range(0, n, stride))
    if len(indices) < 2:
        indices = list(range(n))

    strat_val = initial_capital
    bench_val = initial_capital
    strat_history = [strat_val]
    bench_history = [bench_val]

    rf_step = (1.0 + risk_free_rate_annual) ** (stride / 252.0) - 1.0

    step_returns = []
    trade_outcomes = []

    for idx in indices:
        p = y_prob[idx]
        ret = returns_5d[idx]

        # Benchmark always holds JCI
        bench_val = bench_val * (1.0 + ret)
        bench_history.append(bench_val)

        # Strategy decision
        if p >= threshold:
            step_ret = ret
            strat_val = strat_val * (1.0 + step_ret)
            trade_outcomes.append(1 if ret > 0 else 0)
        else:
            step_ret = rf_step  # Earn risk-free yield while in cash
            strat_val = strat_val * (1.0 + step_ret)

        step_returns.append(step_ret)
        strat_history.append(strat_val)

    strat_curve = np.array(strat_history)
    bench_curve = np.array(bench_history)

    # Returns
    total_ret = ((strat_val - initial_capital) / initial_capital) * 100.0
    bench_ret = ((bench_val - initial_capital) / initial_capital) * 100.0

    # Annualized Return (assuming 252 trading days/year)
    years = max(len(indices) * stride / 252.0, 0.1)
    cagr = ((strat_val / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    # Sharpe Ratio: Annualized
    step_returns_arr = np.array(step_returns)
    excess_returns = step_returns_arr - rf_step
    std_ret = np.std(excess_returns)
    if std_ret > 1e-8:
        steps_per_year = 252.0 / stride
        sharpe = (np.mean(excess_returns) / std_ret) * np.sqrt(steps_per_year)
    else:
        sharpe = 0.0

    # Maximum Drawdown (MDD)
    peaks = np.maximum.accumulate(strat_curve)
    drawdowns = (strat_curve - peaks) / peaks
    mdd = float(np.abs(np.min(drawdowns)) * 100.0)

    # Win rate of active trades
    if trade_outcomes:
        win_rate = (sum(trade_outcomes) / len(trade_outcomes)) * 100.0
    else:
        win_rate = 0.0

    return FinancialMetrics(
        total_return=float(total_ret),
        benchmark_return=float(bench_ret),
        annualized_return=float(cagr),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(mdd),
        win_rate=float(win_rate),
        total_trades=len(trade_outcomes),
        equity_curve=strat_curve,
        benchmark_curve=bench_curve,
>>>>>>> main
    )
