# MSDL-JCI

Thesis research repo: predict JCI (t+5) direction from technical + macro + news via adaptive soft-gating fusion. **Closed as a documented negative result** — the proposed gating model is not viable and nothing is deployable. Start with `README.md`.

This branch (`refactor`) is a mid-migration simplification: there is no `data/`, `docs/`, `scripts/`, `reports/`, `configs/`, `Makefile`, or `Dockerfile` here. Statements below describe what is actually on disk, not the old `main` layout.

## Commands (uv, Python 3.12, uv.lock)

- Setup: `uv sync --extra dev`. `ruff` and `mypy` live in the `dev` optional-dependency, so plain `uv sync` does NOT put them on PATH. `uv.lock` is the single source of truth (`requirements.txt` was removed).
- Test: `uv run pytest -q`. **Currently RED on this branch: 10 collection errors** — `tests/` and `dataset_builder.py` import `Settings`/`get_settings`, but `src/msdl_jci/config/settings.py` only provides the DB-only `Config` (known refactor gap, fix pending).
- Lint: `uv run ruff check .` — advisory only (CI runs it with `--exit-zero`). Known findings: 2× `I001` import-sort plus 2× `F821` (`get_settings` used in `models/macro/encoder.py`, same refactor gap as above). `ruff format` is NOT enforced (16 files would be reformatted).
- Typecheck: `uv run mypy src` — **non-blocking** in CI; only a small subset is checked (`config/`, `evaluation/`, `experiments/`, `utils/`, `models/fusion.py`; most of `models/` and tests are excluded).
- Run: `uv run msdl-jci --mode {proposed,ablation,legacy} --epochs N`, or `uv run msdl-jci-ablation`. (Blocked by the same `get_settings` gap until the config refactor lands.)
- Notebooks: `uv run jupyter execute notebooks/development.ipynb --inplace` (verified clean — 8 code cells, all pass; needs `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`).
- Settings: `src/msdl_jci/config/settings.py` is a minimal DB-only `Config` (`DATABASE_PATH` default `./database/main_database.db`, `LOG_LEVEL` default `INFO`). `.env` and `database/` are git-ignored.

## Data (what is actually here)

- There is **no `data/` directory** on this branch — no raw CSVs, no embeddings, no sample files.
- Tests are fully synthetic (generated in-code). `tests/conftest.py` adds `src/` to `sys.path`.
- `notebooks/development.ipynb` is self-contained: it mirrors the builder logic and runs a synthetic smoke demo (train-only scaling, windowed tensors). No raw files needed.
- `notebooks/01_data_alignment_example.ipynb` and `notebooks/app.ipynb` are **stale here** — they read `data/raw/` and `reports/`/`data/processed/`, which do not exist on this branch.
- The news fallback still applies wherever the builder runs: when `daily_news_embeddings.csv` is absent, that modality is all-zero vectors with no signal.

## Leak-free methodology — do not "fix" these

- Order: align → **train-only scaling** → window (lookback 28) → embargoed walk-forward (70% train, `embargo=5` = label horizon) → AdamW + early stop → validation-threshold → frozen test eval.
- Always pass `train_end_idx` when fitting scalers; fitting on the full frame leaks test ranges (builder warns and `evaluation/walk_forward.py` splits accordingly).
- Macro alignment is ffill-only with leading NaNs dropped — never backfill (`bfill` injects future macro values).
- Target = `future_close > Close` at t+5. Seeds are 42 throughout (`set_all_seeds`).
- `main()` auto-selects `legacy` mode when `sys.argv` contains `pytest` (main.py:93) so the integration test stays fast.

## Simplification notes (this branch)

- Removed: `.idea/` (IDE config), `requirements.txt` (redundant with `uv.lock`), regenerable caches (`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`).
- CI (`.github/workflows/ci.yml`): single `uv sync --frozen --extra dev`; dropped the broken `uv pip install -r .\requirements.txt` step (Windows path, redundant) and the failing `ruff format --check` step. Lint is advisory, mypy non-blocking, pytest reporting.
- Kept (untouched, your call): `.coderabbit.yaml` (untracked bot config), `database/main_database.db` (git-ignored, unused by the pipeline), stale `01_*`/`app.ipynb` notebooks.
