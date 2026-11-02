"""Stats-guard tests: youden_threshold, paired_stats, trading_stats, attach_news.

Extracts the four functions straight from notebooks/development.ipynb (AST,
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
NS = {"np": np, "pd": pd, "roc_curve": roc_curve, "NEWS_COLS": [f"news_{i}" for i in range(768)]}
CELL = 6  # utilities cell in notebooks/development.ipynb
WANT = {"youden_threshold", "paired_stats", "trading_stats", "attach_news"}
with open(ROOT / "notebooks" / "development.ipynb", encoding="utf-8") as f:
    src = "".join(json.load(f)["cells"][CELL]["source"])
mod = ast.parse(src)
picked = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
assert {n.name for n in picked} == WANT, {n.name for n in picked}
exec(compile(ast.Module(body=picked, type_ignores=[]), "<stats>", "exec"), NS)  # noqa: S102
youden_threshold, paired_stats, trading_stats, attach_news = (NS[n] for n in
                                                  ("youden_threshold", "paired_stats", "trading_stats", "attach_news"))


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
    probs = [0.99] * len(dates)
    baseline = trading_stats(dates, probs, df, thr=0.5, hold=5)["strategy"]["daily"]
    probs[10] = 0.1
    out = trading_stats(dates, probs, df, thr=0.5, hold=5)["strategy"]["daily"]
    assert out[10] == baseline[10], out[8:13]  # signal day itself earns nothing yet
    for s in range(11, 16):
        assert abs(out[s] - baseline[s] + 2 * 0.01 / 5) < 1e-12, (s, out[s])
    assert np.allclose(out[16:], baseline[16:]), out[16:20]


def test_down_predictions_short_instead_of_staying_flat():
    df = _flat_market()
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    stats = trading_stats(dates, [0.1] * len(dates), df, thr=0.5, hold=5)["strategy"]
    daily = np.array(stats["daily"])
    assert np.all(daily[5:] < 0.0), daily[5:]
    assert stats["equity"][-1] < stats["equity"][5], stats["equity"][-1]


def test_costs_reduce_strategy_return():
    df = _flat_market()
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    probs = [0.1] * 40
    probs[10] = 0.99
    free = trading_stats(dates, probs, df, cost_bps=0.0)["strategy"]["total_return"]
    paid = trading_stats(dates, probs, df, cost_bps=25.0)["strategy"]["total_return"]
    assert paid < free, (free, paid)


def test_attach_news_vectorized_matches_reference():
    """Verify notebook's attach_news (vectorized) against a reference loop."""
    M, D, N = 10, 768, 5
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((M, D)).astype(np.float32)
    days = pd.date_range("2020-01-01", periods=M, freq="D").values
    news_daily = pd.DataFrame({
        "day": days,
        "vec": [vecs[i] for i in range(M)]
    })
    dates = pd.date_range("2020-01-01", periods=N, freq="D")
    m = pd.DataFrame({"Date": dates})

    # Call the actual attach_news function from the notebook
    result_df = attach_news(m, news_daily)
    
    # Compute expected result via reference loop
    idx = np.searchsorted(days, dates.values, side="left")
    expected_emb = np.zeros((N, D), dtype=np.float32)
    for i, n in enumerate(idx):
        if n > 0:
            expected_emb[i] = vecs[:n].mean(axis=0)

    news_cols = [f"news_{i}" for i in range(768)]
    actual_emb = result_df[news_cols].values.astype(np.float32)
    assert np.allclose(actual_emb, expected_emb, atol=1e-6), "Notebook attach_news output differs from reference loop"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} passed")
    sys.exit(0)