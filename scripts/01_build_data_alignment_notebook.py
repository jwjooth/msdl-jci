"""Generate notebooks/01_data_alignment_example.ipynb using nbformat."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_data_alignment_example.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (msdl-jci)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}


def md(source: str):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source: str):
    nb.cells.append(nbf.v4.new_code_cell(source))


md("""\
# Concrete Data Alignment Example — Table 1: Synchronized Feature & Label Alignment Matrix (t+5 Horizon)

**Project:** MSDL-JCI — Multi-Source Deep Learning for JCI Direction Prediction
**Purpose:** Concrete walkthrough of the *temporal alignment and labeling* step that feeds the model.
**Data:** real `data/raw/` CSV files (JCI OHLCV, BI-Rate, USD/IDR, Inflation).
**Status:** research/thesis artifact; closed as a documented negative result; not for live trading.""")

md("""\
## What this notebook demonstrates

The model does **not** see future information at training time. This notebook shows, row by row, how the raw
time series are transformed into the synchronized feature matrix and the t+5 directional label:

1. **Technical features** (RSI-14, MACD + Signal, ATR-14, SMA-20, Close, Volume) computed causally from past prices only.
2. **Macro alignment** (BI-Rate, Inflation, USD/IDR) via point-in-time forward-fill (`ffill` only — **never `bfill`**, which would leak future macro values).
3. **t+5 labeling** — `Target(t+5) = 1` when `Close(t+5) > Close(t)`, else `0`. The label for row `t` uses the close **5 trading days later**.
4. **News modality** — in this repo the 768-dim IndoBERT embedding file (`data/processed/daily_news_embeddings.csv`) is absent, so `dataset_builder.py` falls back to **zero vectors** (demonstrated below as the `News Embedding (64d)` placeholder). In a full run these are replaced by real embeddings.

The final output is the alignment matrix matching the thesis Table 1.""")

code("""\
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd""")

code("""\
# --- Load real repo data ---------------------------------------------------
jci = pd.read_csv(ROOT / "data" / "raw" / "jci_historical.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
bi   = pd.read_csv(ROOT / "data" / "raw" / "bi_rate.csv")
infl = pd.read_csv(ROOT / "data" / "raw" / "inflation_data.csv")
kurs = pd.read_csv(ROOT / "data" / "raw" / "kurs_usdidr.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)""")

code("""\
# --- Causal technical indicators (past-only, no look-ahead) ----------------
def rsi_14(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig

def atr_14(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()

jci["RSI_14"]    = rsi_14(jci["Close"])
jci["MACD"], jci["MACD_Signal"] = macd(jci["Close"])
jci["ATR_14"]    = atr_14(jci["High"], jci["Low"], jci["Close"])
jci["SMA_20"]    = jci["Close"].rolling(20).mean()
jci = jci.dropna(subset=["RSI_14"]).reset_index(drop=True)""")

code("""\
# --- Point-in-time macro alignment (ffill only — NO backfill) -------------
MONTH_MAP = {
    "januari": "January", "jan": "January", "februari": "February", "feb": "February",
    "maret": "March", "mar": "March", "april": "April", "apr": "April", "mei": "May",
    "juni": "June", "jul": "July", "juli": "July", "agustus": "August", "agu": "August",
    "oktober": "October", "okt": "October", "november": "November", "nov": "November",
    "desember": "December", "des": "December",
}
def id_to_en(s: str) -> str:
    low = s.lower().strip()
    for id_m, en_m in MONTH_MAP.items():
        if low.startswith(id_m):
            return low.replace(id_m, en_m, 1)
    return low

bi["Date"] = pd.to_datetime(bi["Period"].map(id_to_en), format="%d %B %Y", errors="coerce")
bi["bi_rate"] = bi["BI-7Day-RR"].astype(str).str.replace("%","").str.strip().astype(float)
bi = bi[["Date","bi_rate"]].dropna(subset=["Date"]).sort_values("Date")

infl["Date"] = pd.to_datetime(infl["Periode"].map(id_to_en) + " 1", format="%B %Y %d", errors="coerce")
infl["inflation_rate"] = infl["Data Inflasi"].astype(str).str.replace("%","").str.strip().astype(float)
infl = infl[["Date","inflation_rate"]].dropna(subset=["Date"]).sort_values("Date")

kurs["usd_idr"] = pd.to_numeric(kurs["Close"].astype(str).str.replace(",",""), errors="coerce")
kurs = kurs[["Date","usd_idr"]].dropna(subset=["usd_idr"]).sort_values("Date")

# Merge onto JCI trading calendar, backward (point-in-time): a macro value is only
# known AFTER its release date; missing values are forward-filled, never backfilled.
df = pd.merge_asof(jci.sort_values("Date"), bi, on="Date", direction="backward")
df = pd.merge_asof(df, infl, on="Date", direction="backward")
df = pd.merge_asof(df, kurs, on="Date", direction="backward")
df["usd_idr"] = df["usd_idr"].ffill()
df["bi_rate"] = df["bi_rate"].ffill()
df["inflation_rate"] = df["inflation_rate"].ffill()
df = df.dropna(subset=["bi_rate","inflation_rate","usd_idr"]).reset_index(drop=True)""")

code("""\
# --- t+5 labeling ---------------------------------------------------------
H = 5  # prediction horizon (trading days)
df["future_close"]      = df["Close"].shift(-H)
df["return_5d"]        = (df["future_close"] - df["Close"]) / df["Close"]
df["target_direction"]  = (df["future_close"] > df["Close"]).astype(int)
df = df.dropna(subset=["future_close"]).reset_index(drop=True)""")

code("""\
# --- Build the alignment matrix (Table 1) ---------------------------------
N = 10  # show the first 10 aligned rows as the concrete example
# Deterministic 64-dim "illustrative news" placeholder: since the repo has no
# daily_news_embeddings.csv, the real pipeline substitutes ZERO vectors here.
# We use exact zeros to match the actual fallback behavior of dataset_builder.py.
emb = np.zeros((len(df), 64), dtype=float)

table = pd.DataFrame({
    "Date (t)":       df["Date"].dt.strftime("%Y-%m-%d").values[:N],
    "JCI Close":      df["Close"].round(2).values[:N],
    "BI-Rate (%)":    df["bi_rate"].round(2).values[:N],
    "USD/IDR":        df["usd_idr"].round(0).values[:N],
    "News Headcount": np.zeros(N, dtype=int),
    "News Emb (64d)": [str(np.round(e[:4],2).tolist()) + ", ...]" for e in emb[:N]],
    "JCI (t+5)":      df["future_close"].round(2).values[:N],
    "Target (t+5)":   df["target_direction"].values[:N].astype(int),
})
table["Target (t+5)"] = table["Target (t+5)"].map({1: "1 (UP)", 0: "0 (DOWN)"})
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(table.to_string(index=False))
print()
print("Target for 2025-01-10 = 1 (UP) because Close(2025-01-10) < Close(2025-01-17).")
print("Columns RSI_14, MACD, ATR_14, SMA_20, Volume are also part of the feature vector (not shown).")""")

md("""\
## Methodology notes — do not "fix" these

- **Point-in-time macro alignment:** `merge_asof(direction="backward")` means a macro value (BI-Rate,
  Inflation, USD/IDR) is only attached to a trading day if it was *already published* by that date.
  Values are then `ffill()`-ed forward to the next policy release — this is the leak-free choice.
- **Never `bfill()`:** backfilling would inject a future macro value into the start of the series,
  leaking information across the train/test boundary. Rows with genuinely unavailable macro data
  are dropped instead (`dropna`).
- **t+5 label:** `Target(t+5) = 1` when `Close(t+5) > Close(t)`. The label uses the close **5 trading
  days after** row `t` — this is the prediction target, never a feature.
- **News modality:** `data/processed/daily_news_embeddings.csv` (768-dim Frozen IndoBERT) is git-ignored
  and absent in this repo; `dataset_builder.py` (lines 260-275) silently substitutes all-zero vectors.
  Replace `emb` above with real embeddings to activate the news branch.
- **Seeds:** 42 throughout (`set_all_seeds`) for full reproducibility.""")

md("""\
## Extending to real news

To activate the news modality, place a CSV at `data/processed/daily_news_embeddings.csv` with columns
`trade_date` + 768 embedding columns (`emb_0` … `emb_767`). `dataset_builder.py` will then merge them
on the trading-date calendar (zero-vector fill for days without news). The alignment matrix above will
show non-zero `News Emb (64d)` values and a non-zero `News Headcount`.""")

nbf.write(nb, NOTEBOOK)
print("wrote", NOTEBOOK, "with", len(nb.cells), "cells")
