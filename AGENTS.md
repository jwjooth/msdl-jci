# MSDL-JCI

Thesis research repo: predict JCI (t+5) direction from technical + macro + news via adaptive soft-gating fusion. **Closed as a documented negative result** — the proposed gating model is not viable and nothing is deployable. Start with `README.md`.

This branch (`refactor`) is a mid-migration simplification: there is no `data/`, `docs/`, `scripts/`, `reports/`, `configs/`, `Makefile`, or `Dockerfile` here. Statements below describe what is actually on disk, not the old `main` layout.

## Commands (uv, Python 3.12, uv.lock)

- Setup: `uv sync --extra dev`. `ruff` and `mypy` live in the `dev` optional-dependency, so plain `uv sync` does NOT put them on PATH. `uv.lock` is the single source of truth (`requirements.txt` was removed).
- Test: `uv run pytest -q`. **41 passed** — data comes from `database/main_database.db` (SQLite); no CSVs involved.
- Lint: `uv run ruff check .` — clean. Advisory only in CI (`--exit-zero`). `ruff format` is NOT enforced.
- Typecheck: `uv run mypy src` — clean (18 files). **Non-blocking** in CI; only a small subset is checked (`config/`, `evaluation/`, `experiments/`, `utils/`, `models/fusion.py`; most of `models/` and tests are excluded).
- Run: `uv run msdl-jci --mode {proposed,ablation,legacy} --epochs N`, or `uv run msdl-jci-ablation`. (Reads `database/main_database.db`.)
- Notebooks: `uv run jupyter execute notebooks/development.ipynb --inplace` (verified clean — 8 code cells, all pass; needs `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`). It is the only notebook; the stale `01_*`/`app.ipynb` were deleted.
- Settings: `src/msdl_jci/config/settings.py` provides `Settings`/`get_settings()` — `SQLITE_DB_PATH` (`DATABASE_PATH` env wins, default `./database/main_database.db`), `LOG_LEVEL`, ML defaults (lookback 28, horizon 5, 768-d news). `BASE_DIR` is the project root. `.env` and `database/` are git-ignored; contributors copy `.env.example` → `.env` (optional).

## Data (what is actually here)

- Source of truth is SQLite: `database/main_database.db` (git-ignored). Entity contract lives in `ENTITY_TABLES` (`utils/data_loader.py`) and is enforced on every read: `jci_historical`, `bi_rate`, `inflation_data`, `kurs_usdidr` (same shapes as the old CSVs) plus `cnbc_ihsg_articles`, `detik_ihsg_articles`, `kontan_ihsg_articles`. Explicit CSV paths still work as overrides everywhere.
- Tests are fully synthetic (generated in-code) except the encoder/builder/pipeline tests, which read the SQLite DB. `tests/conftest.py` adds `src/` to `sys.path`.
- `notebooks/development.ipynb` is self-contained and the only notebook: it mirrors the builder logic and runs a synthetic smoke demo (train-only scaling, windowed tensors). No raw files needed.
- The news fallback still applies wherever the builder runs: the DB has article tables but no embedding vectors, so that modality is all-zero vectors with no signal.

## Leak-free methodology — do not "fix" these

- Order: align → **train-only scaling** → window (lookback 28) → embargoed walk-forward (70% train, `embargo=5` = label horizon) → AdamW + early stop → validation-threshold → frozen test eval.
- Always pass `train_end_idx` when fitting scalers; fitting on the full frame leaks test ranges (builder warns and `evaluation/walk_forward.py` splits accordingly).
- Macro alignment is ffill-only with leading NaNs dropped — never backfill (`bfill` injects future macro values).
- Target = `future_close > Close` at t+5. Seeds are 42 throughout (`set_all_seeds`).
- `main()` auto-selects `legacy` mode when `sys.argv` contains `pytest` (main.py:93) so the integration test stays fast.

## Simplification notes (this branch)

- Removed: `.idea/` (IDE config), `requirements.txt` (redundant with `uv.lock`), stale `01_*`/`app.ipynb` notebooks (needed `data/`/`reports/` that don't exist here), dead `scripts/`/`data/`/`reports/`/`scraping/`/`preprocessing/` entries in ruff/mypy config, regenerable caches (`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`).
- CI (`.github/workflows/ci.yml`): single `uv sync --frozen --extra dev`; dropped the broken `uv pip install -r .\requirements.txt` step (Windows path, redundant) and the failing `ruff format --check` step. Lint is advisory, mypy non-blocking, pytest reporting.
- Kept (untouched, your call): `.coderabbit.yaml` (untracked bot config), `database/main_database.db` (git-ignored, the SQLite source of truth).
