# 03 — Data Pipeline

1. **Sources.** `data/raw/technical/jci_historical.csv` (OHLCV),
   `data/raw/macro/{bi_rate,inflation_data,kurs_usdidr}.csv`,
   `data/processed/daily_news_embeddings.csv` (768-d IndoBERT).
2. **Technical features.** RSI-14, MACD(12,26,9)+signal, ATR-14, SMA-20
   (`calculate_technical_indicators`). NaN warmup rows dropped.
3. **Macro alignment.** `merge_asof(backward)` + **forward-fill only**; leading
   NaNs dropped (no `bfill` — it leaks future values). Period dates treated as
   available the 1st of next month (conservative; see `reports/phase_04_macro/`).
4. **News alignment.** Exact-date left join; missing → zero vectors (7.1% days).
   No forward-fill, no future dates.
5. **Labels.** t+5: `y = 1{close[t+5] > close[t]}`, `return_5d`, `future_close`
   (never a feature — guarded in code and tests).
6. **Windowing.** Sliding windows lb=28, chronological; tensors
   `(X_tech [N,28,7], X_macro [N,3], X_news [N,768], y, returns, dates)`.
7. **Scaling.** `MinMaxScaler` fit on **train prefix only** (`train_end_idx`).
8. Schemas: `data/schemas/columns.md`; CI sample: `data/sample/aligned_sample.csv`.
