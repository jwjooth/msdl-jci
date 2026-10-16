"""Stats-guard tests: youden_threshold, paired_stats, trading_stats.

Extracts the three functions straight from notebooks/development.ipynb (AST,
no import of torch/heavy deps) and checks them on synthetic inputs.
Run: python tests/test_stats.py (exit 0 = all pass). No pytest needed.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
NS = {"np": np, "pd": pd, "roc_curve": roc_curve}
CELL = 6  # utilities cell in notebooks/development.ipynb

WANT = {"youden_threshold", "paired_stats", "trading_stats"}
src = "".join(json.load(open(ROOT / "notebooks" / "development.ipynb"))["cells"][CELL]["source"])
mod = ast.parse(src)
picked = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
assert {n.name for n in picked} == WANT, {n.name for n in picked}
exec(compile(ast.Module(body=picked, type_ignores=[]), "<stats>"), NS)
youden_threshold, paired_stats, trading_stats = (NS[n] for n in
                                                 ("youden_threshold", "paired_stats", "trading_stats"))


def test_youden_separable():
    yt = np.array([0, 0, 0, 1, 1, 1])
    pt = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = youden_threshold(yt, pt)
    # roc_curve thresholds use >= semantics; thr must separate with margin on the neg side
    assert ((pt >= thr).astype(float) == yt).all(), thr
    assert thr > pt[yt == 0].max(), thr


def test_youden_single_class_falls_back():
    assert youden_threshold(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8])) == 0.5


def test_paired_ci_covers_zero_on_identical_folds():
    a = [0.51, 0.49, 0.52, 0.50, 0.48]
    d, lo, hi = paired_stats(a, list(a))
    assert d == 0.0 and lo <= 0.0 <= hi, (d, lo, hi)


def test_paired_uses_df4_t_value():
    a = [0.6, 0.6, 0.6, 0.6, 0.6]
    b = [0.5, 0.5, 0.5, 0.5, 0.5]
    d, lo, hi = paired_stats(a, b)
    se = np.array([0.1] * 5).std(ddof=1) / np.sqrt(5)
    assert abs(d - 0.1) < 1e-12 and abs((hi - lo) / 2 - 2.776 * se) < 1e-12, (d, lo, hi)


def _flat_market(n=40, ret=0.01):
    dates = pd.bdate_range("2020-01-01", periods=n)
    px = 100.0 * (1 + ret) ** np.arange(n)
    return pd.DataFrame({"Date": dates, "Close": px})


def test_signal_earns_next_hold_days_only():
    df = _flat_market()
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    probs = [0.0] * 40
    probs[10] = 0.99  # single signal at entry 10
    out = trading_stats(dates, probs, df, thr=0.5, hold=5)["strategy"]["daily"]
    assert out[10] == 0.0, out[8:13]  # signal day itself earns nothing yet
    assert out[9] == 0.0, out[8:13]  # pre-signal day must earn nothing (no look-ahead)
    for s in range(11, 16):
        assert abs(out[s] - 0.01 / 5) < 1e-12, (s, out[s])
    assert all(v == 0.0 for v in out[16:]), out[16:20]


def test_costs_widen_the_gap():
    df = _flat_market()
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    probs = [0.0] * 40
    probs[10] = 0.99
    free = trading_stats(dates, probs, df, cost_bps=0.0)["strategy"]["sharpe"]
    paid = trading_stats(dates, probs, df, cost_bps=25.0)["strategy"]["sharpe"]
    assert paid < free, (free, paid)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} passed")
    sys.exit(0)
