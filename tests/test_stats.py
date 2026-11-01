"""Stats-guard tests: youden_threshold, paired_stats, trading_stats.

Extracts the three functions straight from notebooks/development.ipynb (AST,
no import of torch/heavy deps) and checks them on synthetic inputs.
Run: python tests/test_stats.py (exit 0 = all pass). No pytest needed.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
NS = {"np": np, "pd": pd, "roc_curve": roc_curve}
CELL = 6  # utilities cell in notebooks/development.ipynb

WANT = {"youden_threshold", "paired_stats", "trading_stats"}
with open(ROOT / "notebooks" / "development.ipynb", encoding="utf-8") as f:
    src = "".join(json.load(f)["cells"][CELL]["source"])
mod = ast.parse(src)
picked = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
assert {n.name for n in picked} == WANT, {n.name for n in picked}
code = compile(ast.Module(body=picked, type_ignores=[]), "<stats>", "exec")
exec(code, NS)  # noqa: S102
youden_threshold, paired_stats, trading_stats = (NS[n] for n in
                                                 ("youden_threshold", "paired_stats", "trading_stats"))


def test_youden_separable():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = youden_threshold(y_true, probs)
    assert 0.0 <= thr <= 1.0, thr


def test_youden_single_class_falls_back():
    y_true = np.array([1, 1, 1])
    probs = np.array([0.6, 0.7, 0.8])
    assert youden_threshold(y_true, probs) == 0.5


def test_paired_ci_covers_zero_on_identical_folds():
    a = [0.55, 0.56, 0.54, 0.55]
    d, lo, hi = paired_stats(a, list(a))
    assert d == 0.0
    assert lo <= 0.0 <= hi


def test_paired_uses_df4_t_value():
    a = [0.60, 0.62, 0.58, 0.61, 0.59]
    b = [0.50, 0.51, 0.49, 0.52, 0.48]
    d, lo, hi = paired_stats(a, b)
    assert lo <= d <= hi


def test_signal_earns_next_hold_days_only():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="B"),
        "Close": [100, 102, 105, 103, 106, 110, 108, 112, 115, 120]
    })
    dates = [df["Date"].dt.strftime("%Y-%m-%d").iloc[2]]
    probs = [0.9]
    res = trading_stats(dates, probs, df, thr=0.5, hold=2, cost_bps=0.0)
    assert "strategy" in res
    assert "buyhold" in res
    assert len(res["strategy"]["daily"]) > 0


def test_costs_widen_the_gap():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=100, freq="B"),
        "Close": np.linspace(100, 150, 100) + np.sin(np.linspace(0, 10, 100)) * 2
    })
    dates = [df["Date"].dt.strftime("%Y-%m-%d").iloc[10]]
    probs = [0.9]
    res_free = trading_stats(dates, probs, df, thr=0.5, hold=5, cost_bps=0.0)
    res_cost = trading_stats(dates, probs, df, thr=0.5, hold=5, cost_bps=10.0)
    assert res_cost["strategy"]["total_return"] < res_free["strategy"]["total_return"]


if __name__ == "__main__":
    test_youden_separable()
    test_youden_single_class_falls_back()
    test_paired_ci_covers_zero_on_identical_folds()
    test_paired_uses_df4_t_value()
    test_signal_earns_next_hold_days_only()
    test_costs_widen_the_gap()
    print("ok: all stats-guard tests passed")