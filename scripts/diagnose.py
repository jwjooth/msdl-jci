#!/usr/bin/env python3
# ruff: noqa: S102, SIM115
"""Diagnostic script for MSDL-JCI notebook. Prints data pipeline and split diagnostics."""
# ruff: noqa: S102, SIM115
import json
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NB_ROOT))

# Read the notebook and extract needed cells
nb = json.load(open(NB_ROOT / 'notebooks' / 'development.ipynb'))
imports_src = ''.join(nb['cells'][1]['source'])
entity_src = ''.join(nb['cells'][4]['source'])
utils_src = ''.join(nb['cells'][6]['source'])

# Create a module namespace
import types

mod = types.ModuleType('notebook_ns')

# Execute imports cell first (defines CFG, USE_DB, etc.)
exec(imports_src.replace('%matplotlib inline', ''), mod.__dict__)

# Execute entity cell (defines ENTITY_TABLES, read_table, USE_DB)
exec(entity_src, mod.__dict__)

# Execute utils cell
exec(utils_src, mod.__dict__)

CFG = mod.CFG
load_frames = mod.load_frames
align = mod.align
make_tensors = mod.make_tensors
USE_DB = mod.USE_DB

print("=== Data Pipeline Diagnostics ===")

jci, bi, inf, kur = load_frames()
print(f"jci_historical: {len(jci)} rows, date range: {jci['Date'].min()} → {jci['Date'].max()}")
print(f"bi_rate: {len(bi)} rows, periods: {bi['Period'].iloc[0]} → {bi['Period'].iloc[-1]}")
print(f"inflation_data: {len(inf)} rows, periods: {inf['Periode'].iloc[0]} → {inf['Periode'].iloc[-1]}")
print(f"kurs_usdidr: {len(kur)} rows, date range: {kur['Date'].min()} → {kur['Date'].max()}")

print(f"\nBI rate first period: {bi['Period'].iloc[0]}")
print(f"Inflation first period: {inf['Periode'].iloc[0]}")

df = align(jci, bi, inf, kur)
print(f"\nAfter align(): {len(df)} rows")
print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")

# Check for dropped rows
print("\nRows dropped in alignment:")
print("  - NaN in tech indicators (RSI/MACD/ATR/SMA warmup): ~35")
print(f"  - NaN in macro cols: {df[['bi_rate','inflation_rate','usd_idr']].isna().sum().sum()}")
print(f"  - NaN in future_close: {df['future_close'].isna().sum()}")

# Check label NaN issue (D2)
fc = df['Close'].shift(-CFG.horizon)
nan_labels = (fc.isna()).sum()
print(f"\nNaN labels (rows where t+5 has no valid future close): {nan_labels}")

# Check zero news fraction
zero_news = (df[[f'emb_{i}' for i in range(CFG.projection_dim)]] == 0).all(axis=1).sum()
print(f"Trading days with zero news (all-zero emb): {zero_news} / {len(df)} ({zero_news/len(df):.1%})")

# Check macro availability
print("\nMacro first valid row indices:")
print(f"  bi_rate: index {df['bi_rate'].first_valid_index()}")
print(f"  inflation_rate: index {df['inflation_rate'].first_valid_index()}")
print(f"  usd_idr: index {df['usd_idr'].first_valid_index()}")

# Check make_tensors
lb = CFG.look_back
n_win = len(df) - lb + 1
train_end_df = int(n_win * 0.70) + lb - 1
X_tech, X_macro, X_news, y, dates, feat_scaler = make_tensors(df, train_end_df)
print(f"\nTensor shapes: X_tech={X_tech.shape}, X_macro={X_macro.shape}, X_news={X_news.shape}, y={y.shape}")
print(f"Sample count: {len(y)} (expected: {len(df) - lb + 1} = {len(df)} - {lb} + 1)")
print(f"Horizon NOT subtracted: {len(y) == len(df) - lb + 1}")

# Check scaler
raw_tech = df[CFG.tech_cols].values.astype(np.float32)
print("\nScaler fit check:")
print(f"  feat_scaler.data_min_ = {feat_scaler.data_min_}")
print(f"  raw_tech[:train_end_df].min(axis=0) = {raw_tech[:train_end_df].min(axis=0)}")
print(f"  Match: {np.allclose(feat_scaler.data_min_, raw_tech[:train_end_df].min(axis=0))}")

# Check fraction of test outside train range
test_tech = raw_tech[train_end_df:]
train_min = raw_tech[:train_end_df].min(axis=0)
train_max = raw_tech[:train_end_df].max(axis=0)
out_of_range = ((test_tech < train_min) | (test_tech > train_max)).mean()
print(f"  Fraction of test features outside train range: {out_of_range:.1%}")

# Walk-forward split analysis
print("\n=== Walk-Forward Split Analysis ===")
total_len = len(df)
n_splits = 5
for i in range(n_splits):
    train_end = total_len - (n_splits - i) * (total_len // n_splits // 2)
    train_end = min(train_end, total_len - CFG.horizon - CFG.look_back)
    if train_end < CFG.look_back + CFG.horizon:
        print(f"  Fold {i}: SKIPPED (train_end={train_end} < {CFG.look_back + CFG.horizon})")
        continue
    hs, he = train_end - CFG.look_back, min(train_end - CFG.look_back + CFG.horizon, total_len - 1)
    test_size = he - hs
    print(f"  Fold {i}: train_end={train_end}, test indices=[{hs}:{he}], test_size={test_size}")
    print(f"    Train dates: {dates[0]} → {dates[train_end-CFG.look_back-1] if train_end-CFG.look_back-1 >= 0 else 'N/A'}")
    print(f"    Test dates: {dates[hs]} → {dates[he-1] if he-1 < len(dates) else 'N/A'}")

# Gate analysis on a fixed batch
print("\n=== Gate Analysis (fixed batch) ===")
if hasattr(mod, '_HAS_TORCH') and mod._HAS_TORCH:
    import torch
    model = mod.StockModel(CFG)
    model.eval()
    with torch.no_grad():
        tb = torch.tensor(X_tech[:1], dtype=torch.float32)
        mb = torch.tensor(X_macro[:1], dtype=torch.float32)
        logits, loss, weights = model(tb, mb, torch.zeros(1, 512, dtype=torch.long))
        print(f"  Gate weights (post-softmax): α={weights[0,0]:.4f}, β={weights[0,1]:.4f}, γ={weights[0,2]:.4f}")
        print(f"  Sum of weights: {weights.sum():.4f}")
        
        _, (hn, _) = model.lstm(tb)
        tech_feat = hn.squeeze(0)
        macro_raw = model.mlp(mb)
        news_feat = model.news_proj(torch.randn(1, 768) * 0.1)
        print(f"  Tech branch: mean={tech_feat.mean():.4f}, std={tech_feat.std():.4f}")
        print(f"  Macro branch: mean={macro_raw.mean():.4f}, std={macro_raw.std():.4f}")
        print(f"  News branch: mean={news_feat.mean():.4f}, std={news_feat.std():.4f}")

print("\n=== Diagnosis complete ===")
