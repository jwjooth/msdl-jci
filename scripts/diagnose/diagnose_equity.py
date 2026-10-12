"""Phase-1 diagnosis (READ-ONLY): why is the strategy equity curve flat?
Reads stored notebook outputs + rebuilds price/return frame from SQLite.
Never touches training/eval code. Run: python scripts/diagnose/diagnose_equity.py
"""
import json
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

# ---- 1. stored arrays from notebook (full variant only is stored) ----
nb = json.load(open(ROOT / "notebooks" / "development.ipynb"))
wf = None
for c in nb["cells"]:
    if c.get("id") == "run":
        pass
# wf_metrics lives only in kernel memory; recover what cells printed/stored:
# fold_metrics preds/truths are stored in outputs? No — recover from `ablation` is summary-only.
# So: rebuild preds is impossible without torch. Instead use what IS stored:
# -> per-fold auc/acc/f1/thr/n from viz3 printed table (hard numbers from frozen run).
FOLDS = [
    {"fold": 0, "auc": 0.5007, "acc": 0.4803, "f1": 0.1222, "n": 304},
    {"fold": 1, "auc": 0.5178, "acc": 0.4408, "f1": 0.0000, "n": 304},
    {"fold": 2, "auc": 0.4271, "acc": 0.4737, "f1": 0.6117, "n": 304},
    {"fold": 3, "auc": 0.5828, "acc": 0.4211, "f1": 0.1200, "n": 304},
    {"fold": 4, "auc": 0.6018, "acc": 0.5033, "f1": 0.2705, "n": 304},
]
GATES = [  # (alpha, beta, gamma) per fold, viz4 printed table
    (0.0006, 0.9994, 0.0000), (0.0411, 0.9464, 0.0125), (0.4270, 0.4297, 0.1433),
    (0.5683, 0.4175, 0.0141), (0.7248, 0.2725, 0.0027),
]

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

# test entries: last 5 folds x 304 of windowed samples; window lb=28 maps sample i -> df row i+27.
# walk-forward test spans the tail; recover entry dates as last 1520 sample dates of the aligned frame.
# aligned df drops indicator-NaN head (~20 rows); approximate: use tail of px shifted by indicator warmup.
# EXACT mapping needs the notebook df; we verify continuity + magnitudes, not exact indices.
test_px = px.tail(1520).reset_index(drop=True)
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
print("\nstored per-fold metrics (H2: threshold misfit visible in fold1 F1=0.000):")
for m in FOLDS:
    print(f"  fold {m['fold']}: AUC={m['auc']:.4f} Acc={m['acc']:.4f} F1={m['f1']:.4f} n={m['n']}")
print("stored gate weights (H7: gamma max 0.143, never dominant; alpha rises to 0.72 in fold 4):")
for k, (a, b, g) in enumerate(GATES):
    print(f"  fold {k}: alpha={a:.4f} beta={b:.4f} gamma={g:.4f}")

print("\nUNVERIFIABLE without torch/checkpoints: stored per-entry prob arrays, lstm-vs-full pred equality, prob histograms.")
print("To verify H1/H2/H8 fully, run cells viz9/viz12 outputs already pasted in chat (mean pred 0.4482, UP recall 0.1981).")
print("DONE")
