# MSDL-JCI

Thesis research repo: predict JCI (t+5) direction from technical + macro + news.
Implements the hybrid architecture from the thesis defense: **LSTM (technical) + MLP Encoder (macro) + Frozen IndoBERT (news) + Adaptive MLP Fusion (Soft Gating Network)**.

This repo is **one notebook**: `notebooks/development.ipynb` holds config, the SQLite entity contract, all utilities, the model architecture, the run, visualizations, and checks. There is no `src/`, `tests/`, `.github/`, `data/`, `scripts/`, `reports/`, `configs/`, `Makefile`, or `Dockerfile`.

## Commands (uv, Python 3.12, uv.lock)

- Setup: `uv sync --extra dev` (ruff is the only dev dep; `uv.lock` is the single source of truth).
- Run: `uv run jupyter execute notebooks/development.ipynb --inplace` (verified clean — 0 errors; needs `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`). Reads `database/main_database.db` (tracked in git) when present, else a synthetic seed-42 fallback. First run downloads `indobenchmark/indobert-base-p1` and caches article vectors to `database/news_emb_cache.npz`.
- Lint: `uv run ruff check .` — clean. `ruff format` is NOT enforced.

## Settings

The notebook's `Config` cell owns hyperparameters (Table 5). `DB_PATH` is direct (`database/main_database.db`, resolved from repo root) — no env vars. ML defaults: lookback 28, horizon 5, seed 42, projection_dim 64, lstm_dim 64, hidden_layers (32, 16), fusion_output_dim 3. `database/` is tracked in git.

**New dependencies**: `torch>=2.4.0`, `transformers>=4.41.0` for the thesis model architecture.

## Data (what is actually here)

- Source of truth is SQLite: `database/main_database.db` (git-ignored). Entity contract lives in `ENTITY_TABLES` (notebook cell) and is enforced on every read: `jci_historical`, `bi_rate`, `inflation_data`, `kurs_usdidr` plus `cnbc_ihsg_articles`, `detik_ihsg_articles`, `kontan_ihsg_articles`.
- News signal: all 3 portals embedded once with frozen `indobenchmark/indobert-base-p1` CLS (5067 articles parsed incl. Indonesian month names), cached to `database/news_emb_cache.npz`, attached past-only (strictly-before-date means, weekend Sat/Sun → Monday per thesis). 4192 articles fall on/before JCI end (2025-12) and inform training; 875 CNBC/Kontan articles dated 2026 are embedded but cannot inform t+5 labels (no future prices) — excluded by construction, counts printed in the run cell.

## Current result: weak signal, honestly trained

Last executed run trains per fold (fresh model, Adam, weighted BCE, early stopping max100/patience8, `TimeSeriesSplit` n=5, n=304/fold) on real data (1852 rows, 2018–2025). Macro uses Z-score per thesis §2.2 (train-fit only):

- Walk-forward: pooled AUC ≈ 0.51, acc ≈ 0.48, F1 ≈ 0.46; per-fold AUC 0.43–0.61 (mean ≈ 0.53).
- Ablation (pooled AUC): lstm 0.50, lstm_macro 0.51, lstm_news 0.50, static 0.50, full 0.51 — no variant beats chance; macro+news add ~nothing.
- Trading sim (prob>0.5 → hold 5d, no costs): full Sharpe ≈ 0.40 vs buy-hold ≈ 0.39, MDD ≈ -0.40 — no edge.
- Both classes predicted: UP recall ≈ 0.40, DOWN recall ≈ 0.59 at threshold 0.5 — no more constant-UP collapse.
- Gating alive and varying: α (tech) ≈ 0.20–0.64, β (macro) ≈ 0.36–0.63, γ (news) ≈ 0.00–0.19 — news carries ~nothing.

Do not treat these numbers as a bug in the architecture; the leak-free checks cell passes.

## Leak-free methodology — do not "fix" these

- Order: align → **train-only scaling** → window (lookback 28) → t+5 labels.
- Scalers fit on the 70% train prefix only (`train_end_idx`); the checks cell proves it (`data_min_` equals the train-prefix min).
- Macro alignment is ffill-only with leading NaNs dropped — never backfill (`bfill` injects future macro values).
- Target = `future_close > Close` at t+5. Seed is 42 throughout.

## Model architecture (thesis Table 5)

- **Technical branch**: LSTM (hidden=64, lookback=28, dropout=0.2) over technical indicators
- **Macro branch**: MLP Encoder (2 layers: 3→32→16, ReLU) — no lookback, single timestep
- **News branch**: Frozen IndoBERT (`indobenchmark/indobert-base-p1`, CLS precomputed + cached) → Linear projection (768→64)
- **Adaptive MLP Fusion (Soft Gating Network)**: Linear(144→3) + Softmax → weights (α, β, γ)
- **Head**: Linear(64→1) for binary classification (BCEWithLogitsLoss)

## Evaluation (thesis methodology)

- Walk-forward expanding window validation (5 splits)
- Metrics: AUC, Accuracy, F1-Score (binary classification)
- Per-fold metrics bar chart, gating-weights (α, β, γ) evolution + distribution, ROC curves, confusion matrices, prediction-vs-actual timeline, confidence histogram

## Known notebook warts

- Cells 2 and 3 are a duplicated "Config & hyperparameters" markdown cell (identical text) — safe to delete one.
- Cell 0 claims AUC/F1/Sharpe metrics; only AUC/Accuracy/F1 are computed.
- LSTM emits a dropout/num_layers=1 UserWarning on every run (dropout=0.2 with a single-layer LSTM) — harmless, by design.

