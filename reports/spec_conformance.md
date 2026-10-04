# Spec Conformance Checklist

| Thesis Item | Status | Evidence |
|---|---|---|
| Binary direction prediction of JCI, horizon t+5 | IMPLEMENTED | `align()` creates `target_direction = (fc > Close).astype(float)` |
| Target_t = 1 if Close[t+5] > Close[t] | IMPLEMENTED | Same as above |
| Master calendar = JCI trading days | IMPLEMENTED | `align()` uses `jci` dates as master |
| Data span 2018-2025 | DIVERGES | Actual: 2018-05-31 → 2025-12-19 (missing ~5 months) |
| Technical: JCI OHLCV daily | IMPLEMENTED | `load_frames()` reads `jci_historical` |
| Macro: BI-7Day RR, CPI inflation, USD/IDR | IMPLEMENTED | `load_frames()` reads all three |
| News: detik, CNBC, Kontan filtered by "IHSG" | MISSING | No article tables loaded in `load_frames()` |
| RSI, MACD+signal, ATR, SMA, Close, Volume | IMPLEMENTED | `add_indicators()` |
| Min-Max scaling fit on TRAIN only | IMPLEMENTED | `make_tensors()` fits on `[:train_end_idx]` |
| Macro Z-score fit on TRAIN only | MISSING | Uses MinMaxScaler, not Z-score |
| Monthly macro forward-filled to daily | PARTIAL | Uses merge_asof backward (period date, not publication date) |
| News: clean text, IndoBERT tokenizer, max_len 512 | MISSING | No news processing |
| Frozen IndoBERT-base, mean-pool, 768->64 | MISSING | No news embeddings |
| Weekend articles -> Monday | MISSING | No news processing |
| No-news days -> 64-d zero vector | IMPLEMENTED | `align()` adds `emb_0..emb_63 = 0.0` |
| LSTM hidden=64, lookback=28, dropout=0.2 | IMPLEMENTED | `StockModel.__init__` |
| MLP 3->32->16, ReLU, no lookback | IMPLEMENTED | `StockModel.__init__` |
| Frozen IndoBERT(768) + Linear(768->64) | PARTIAL | IndoBERT lazy-loaded but gated; falls back to random |
| Fusion: Linear(144->3) + Softmax -> (α,β,γ) | IMPLEMENTED | `StockModel.forward` |
| Head: Linear(64->1) with BCEWithLogitsLoss | IMPLEMENTED | `StockModel.__init__` |
| How α/β/γ applied to branches of 64/16/64 | DIVERGES | Code does `macro_proj(16->64)` then weighted sum, but `macro_raw` (16-d) is used in gate input, thesis gap |
| Expanding-window walk-forward | PARTIAL | `walk_forward_evaluate` exists but fold sizes are broken (N=1,2,3,4,5) |
| Baseline: pure LSTM on technical only | MISSING | No baseline implemented |
| Ablation: 5 configs | MISSING | No ablation |
| Early stopping | MISSING | No training loop |
| Metrics: Accuracy, F1, Precision, Recall, ROC-AUC, confusion | PARTIAL | Has Accuracy, F1, AUC, confusion; missing Precision, Recall as separate |
| Sharpe, MDD, win rate, cumulative returns | MISSING | No trading simulation |
| Trading simulation vs buy-and-hold vs baseline | MISSING | No trading simulation |
| Gate collapse | CONFIRMED | α=0, β=0, γ=1.0, std=0 across all folds |
| No purge/embargo between train and test | CONFIRMED | No gap in `walk_forward_evaluate` |
| NaN labels silently set to DOWN | CONFIRMED | `(NaN > Close).astype(float)` = 0.0 |
| Per-fold AUC undefined for N=1 | CONFIRMED | Fold 0 AUC=0.5 (silently defaults) |
