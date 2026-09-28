# MSDL-JCI

Thesis research repo: predict JCI (t+5) direction from technical + macro + news. 
Implements the hybrid architecture from the thesis defense: **LSTM (technical) + MLP Encoder (macro) + Frozen IndoBERT (news) + Adaptive MLP Fusion (Soft Gating Network)**.

This repo is **one notebook**: `notebooks/development.ipynb` holds config, the SQLite entity contract, all utilities, the model architecture, the run, visualizations, and checks. There is no `src/`, `tests/`, `.github/`, `data/`, `scripts/`, `reports/`, `configs/`, `Makefile`, or `Dockerfile`.

## Commands (uv, Python 3.12, uv.lock)

- Setup: `uv sync --extra dev` (ruff is the only dev dep; `uv.lock` is the single source of truth).
- Run: `uv run jupyter execute notebooks/development.ipynb --inplace` (verified clean — 8 code cells, all pass; needs `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`). Reads `database/main_database.db` when present, else a synthetic seed-42 fallback.
- Lint: `uv run ruff check .` — clean. `ruff format` is NOT enforced.

## Settings

The notebook's `Config` cell owns hyperparameters (Table 5). `DATABASE_PATH` env var wins, default `./database/main_database.db`. ML defaults: lookback 28, horizon 5, seed 42, projection_dim 64, lstm_dim 64, hidden_layers (32, 16), fusion_input_dim 144, fusion_output_dim 3. `.env` and `database/` are git-ignored; copy `.env.example` → `.env` only if you need to override.

**New dependencies**: `torch>=2.4.0`, `transformers>=4.41.0` for the thesis model architecture.

## Data (what is actually here)

- Source of truth is SQLite: `database/main_database.db` (git-ignored). Entity contract lives in `ENTITY_TABLES` (notebook cell) and is enforced on every read: `jci_historical`, `bi_rate`, `inflation_data`, `kurs_usdidr` plus `cnbc_ihsg_articles`, `detik_ihsg_articles`, `kontan_ihsg_articles`.
- The news fallback still applies: the DB has article tables but no embedding vectors, so that modality is all-zero vectors with no signal.

## Leak-free methodology — do not "fix" these

- Order: align → **train-only scaling** → window (lookback 28) → t+5 labels.
- Scalers fit on the 70% train prefix only (`train_end_idx`); the checks cell proves it (`data_min_` equals the train-prefix min).
- Macro alignment is ffill-only with leading NaNs dropped — never backfill (`bfill` injects future macro values).
- Target = `future_close > Close` at t+5. Seed is 42 throughout.

## Model architecture (thesis Table 5)

- **Technical branch**: LSTM (hidden=64, lookback=28, dropout=0.2) over technical indicators
- **Macro branch**: MLP Encoder (2 layers: 3→32→16, ReLU) — no lookback, single timestep
- **News branch**: Frozen IndoBERT (`csebuetnlp/mubi-bert-base`) → Linear projection (768→64)
- **Adaptive MLP Fusion (Soft Gating Network)**: Linear(144→3) + Softmax → weights (α, β, γ)
- **Head**: Linear(64→1) for binary classification (BCEWithLogitsLoss)

## Evaluation (thesis methodology)

- Walk-forward expanding window validation (5 splits)
- Metrics: AUC, Accuracy, F1-Score (binary classification)
- Baseline comparison: pure LSTM on technical features only
- Ablation: macro-only, news-only, static fusion vs adaptive gating