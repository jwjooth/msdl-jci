"""Phase-1 diagnosis (READ-ONLY): strategy vs buy-hold equity construction.
Reads committed database/wf_artifacts.npz + rebuilds price/return frame from SQLite.
Never touches training/eval code. Run: python scripts/diagnose/diagnose_equity.py
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# ---- 1. stored arrays from the notebook run (committed wf_artifacts.npz) ----
# preds/truths/test_dates per variant — recompute fold metrics from these, never hardcode.
af = np.load(ROOT / "database" / "wf_artifacts.npz", allow_pickle=True)
P = np.asarray(af["full_preds"], dtype=float)
T = np.asarray(af["full_truths"], dtype=float)
D = [str(d) for d in af["full_test_dates"]]
N_FOLDS = 5  # matches the notebook's TimeSeriesSplit(n_splits=5)
assert len(P) == len(T) == len(D) and len(P) % N_FOLDS == 0, (len(P), len(T), len(D))
PER = len(P) // N_FOLDS  # equal fold sizes; derived, not hardcoded
FOLDS = []
for k in range(N_FOLDS):
    p, t = P[k * PER:(k + 1) * PER], T[k * PER:(k + 1) * PER]
    pred05 = (p > 0.5).astype(float)
    FOLDS.append({"fold": k, "auc": float(roc_auc_score(t, p)),
                  "acc": float(accuracy_score(t, pred05)),
                  "f1": float(f1_score(t, pred05, zero_division=0)), "n": len(t)})
print(f"full-variant artifacts: n={len(P)}, test window {D[0]} -> {D[-1]}")

# ---- 2. rebuild price/return frame from DB (news-independent part of align) ----
con = sqlite3.connect(ROOT / "database" / "main_database.db")
jci = pd.read_sql_query("SELECT * FROM [jci_historical]", con)
kur = pd.read_sql_query("SELECT * FROM [kurs_usdidr]", con)
jci["Date"] = pd.to_datetime(jci["Date"], format="mixed", errors="coerce")
jci = jci.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
jci["Close"] = pd.to_numeric(jci["Close"].astype(str).str.replace(",", "", regex=False), errors="coerce")
H = 5
fc = jci["Close"].shift(-H)
jci["return_5d"] = (fc - jci["Close"]) / jci["Close"]
jci["target"] = (fc > jci["Close"]).astype(float)
jci["ret_1d"] = jci["Close"].pct_change()
px = jci.dropna(subset=["return_5d"]).reset_index(drop=True)
print(f"price frame: {len(px)} rows, {px['Date'].min().date()} -> {px['Date'].max().date()}")
print(f"label UP rate: {px['target'].mean():.4f}")

# test entries: last N_FOLDS x PER windowed samples; window lb=28 maps sample i -> df row i+27.
# walk-forward test spans the tail; recover entry dates as last len(P) sample dates of the aligned frame.
# aligned df drops indicator-NaN head (~20 rows); approximate: use tail of px shifted by indicator warmup.
# EXACT mapping needs the notebook df; we verify continuity + magnitudes, not exact indices.
test_px = px.tail(len(P)).reset_index(drop=True)
print(f"test window approx: {test_px['Date'].min().date()} -> {test_px['Date'].max().date()}, n={len(test_px)}")
gaps = test_px["Date"].diff().dt.days
print(f"date gaps >5d inside test window: {(gaps > 5).sum()} (fold-boundary discontinuity check)")

# ---- 3. equity recomputation: overlapping (code as-is) vs correct alternatives ----
r = test_px["return_5d"].values
r1 = test_px["ret_1d"].fillna(0).values
# A) overlapping 5d cumprod (what trading_stats does for buy-hold)
eq_overlap = np.cumprod(1 + r)
# B) proper daily compounding over same window
eq_daily = np.cumprod(1 + r1) / np.cumprod(1 + r1)[0] * 1.0
eq_daily = np.cumprod(1 + r1)
# C) non-overlapping 5d (every 5th entry)
eq_nonoverlap = np.cumprod(1 + r[::5])
print(f"buy-hold terminal: overlapping-5d={eq_overlap[-1]:.3f} | daily={eq_daily[-1]:.3f} | nonoverlap-5d={eq_nonoverlap[-1]:.3f}")
print(f"buy-hold min: overlapping={eq_overlap.min():.3f} | daily={eq_daily.min():.3f}")
print(f"proper JCI total return first->last test window (daily): {eq_daily[-1]-1:+.3f}")
# leverage factor: overlapping compounds ~5x the exposure
print(f"sum of overlapping 5d returns: {np.nansum(r):+.3f} vs sum of daily: {np.nansum(r1):+.3f} (~{np.nansum(r)/np.nansum(r1):.1f}x)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(eq_overlap, label=f"overlapping 5d cumprod (code as-is, end={eq_overlap[-1]:.2f})", color="red")
ax.plot(eq_daily, label=f"daily cumprod (correct, end={eq_daily[-1]:.2f})", color="steelblue")
ax.set_title("Buy-hold equity: overlapping-5d vs daily compounding (same window)")
ax.set_xlabel("test entry"); ax.set_ylabel("portfolio value (start=1)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "buyhold_overlap_vs_daily.png")
print("saved out/buyhold_overlap_vs_daily.png")

# ---- 4. single-trade jump analysis: largest one-step gains in overlapping space ----
jr = pd.Series(r)
top = jr.nlargest(8)
print("largest single-entry 5d returns in test window:")
for idx, v in top.items():
    print(f"  entry {idx} date {test_px['Date'].iloc[idx].date()} r5d={v:+.4f}")
# longest run of consecutive +10% entries (explains multi-step climb without any single huge trade)
hot = (jr.fillna(0) > 0.10).values
best, cur, bi_ = 0, 0, 0
for k, h in enumerate(hot):
    cur = cur + 1 if h else 0
    if cur > best: best, bi_ = cur, k
print(f"longest run of consecutive entries with r5d>+10%: {best} ending at entry {bi_} ({test_px['Date'].iloc[bi_].date()})")

# ---- 5. stored per-fold metric table + gate table (evidence, no retraining) ----
print("\nstored per-fold metrics @0.5 (recomputed from wf_artifacts.npz):")
for m in FOLDS:
    print(f"  fold {m['fold']}: AUC={m['auc']:.4f} Acc={m['acc']:.4f} F1={m['f1']:.4f} n={m['n']}")

print("\nGate weights / prob histograms live in notebook outputs only (need torch to rebuild).")
print("DONE")
