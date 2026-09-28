# MSDL-JCI

Thesis research repo: predict JCI (t+5) direction from technical + macro + news. **Closed as a documented negative result** — nothing is deployable. Start with `README.md`.

This repo is **one notebook**: `notebooks/development.ipynb` holds config, hyperparameters, the SQLite entity contract, all utilities, the run, visualizations, and checks. There is no `src/`, `tests/`, `data/`, `docs/`, `scripts/`, `reports/`, `configs/`, `Makefile`, or `Dockerfile`.

## Commands (uv, Python 3.12, uv.lock)

- Setup: `uv sync --extra dev`. `ruff` is the only dev dependency. `uv.lock` is the single source of truth.
- Run: `uv run jupyter execute notebooks/development.ipynb --inplace` (verified clean — 8 code cells, all pass; needs `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`). Reads `database/main_database.db` when present, else a synthetic seed-42 fallback.
- Lint: `uv run ruff check .` — clean. `ruff format` is NOT enforced.
- Settings live in the notebook's `Config` (`DATABASE_PATH` env wins, default `./database/main_database.db`; ML defaults: lookback 28, horizon 5, seed 42, 768-d news). `.env` and `database/` are git-ignored; contributors copy `.env.example` → `.env` (optional).

## Data (what is actually here)

- Source of truth is SQLite: `database/main_database.db` (git-ignored). Entity contract lives in `ENTITY_TABLES` (notebook cell) and is enforced on every read: `jci_historical`, `bi_rate`, `inflation_data`, `kurs_usdidr` plus `cnbc_ihsg_articles`, `detik_ihsg_articles`, `kontan_ihsg_articles`.
- The news fallback still applies: the DB has article tables but no embedding vectors, so that modality is all-zero vectors with no signal.

## Leak-free methodology — do not "fix" these

- Order: align → **train-only scaling** → window (lookback 28) → t+5 labels.
- Scalers fit on the 70% train prefix only (`train_end_idx`); the checks cell proves it (`data_min_` equals the train-prefix min).
- Macro alignment is ffill-only with leading NaNs dropped — never backfill (`bfill` injects future macro values).
- Target = `future_close > Close` at t+5. Seed is 42 throughout.
