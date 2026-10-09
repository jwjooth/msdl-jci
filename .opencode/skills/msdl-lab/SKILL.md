---
name: msdl-lab
description: Run or modify the MSDL-JCI thesis notebook without breaking leak-free methodology. Use for any experiment, ablation, or metric change in notebooks/development.ipynb.
---

# MSDL Lab — experiment guardrails

Single notebook repo: `notebooks/development.ipynb` (config → utils → model → run → viz → checks). `Config` cell owns all hyperparameters (Table 5: lookback 28, horizon 5, seed 42).

## Hard rules (from AGENTS.md — never loosen to "improve" metrics)

- Order: align → train-only scaling (70% prefix, `train_end_idx`) → window → t+5 labels (`future_close > Close`).
- Macro: ffill-only, drop leading NaNs. Never `bfill`.
- News: frozen `indobenchmark/indobert-base-p1` CLS via `database/news_emb_cache.npz`, past-only attach (Sat/Sun → Monday). Since 2026-10-09 all 3 portals inform training (detik 4953 + cnbc 5556 + kontan 10275, merged from `database/new_database.db`).
- All SQLite reads through `read_table` (`ENTITY_TABLES` contract).
- Missing DB → synthetic seed-42 fallback; fallback numbers are meaningless, never quote them.

## Result being defended

Honest weak signal: pooled AUC ≈ 0.50, no variant/trading edge, news gate γ ≈ 0.03. Checks cell must pass. Keep BCE, lookback 28, LR 1e-3 (focal/grid were negative, #38).

## Baselines to preserve

Ablation `StockModel.VARIANTS`: `lstm`, `lstm_macro`, `lstm_news`, `static`, `full` + E1 `lstm_news_shuffled`. `v1.0-bab4` is the frozen Ch.4 run — never quote post-tag numbers as Bab-4 results.
