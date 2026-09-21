# MSDL-JCI

Thesis research repo: predict JCI (t+5) direction from technical + macro + news via adaptive soft-gating fusion. **Closed as a documented negative result** — the proposed gating model is not viable and nothing is deployable. Start with `README.md` (narrative) and `docs/02_architecture.md` (module map).

## Commands (uv, Python 3.12, uv.lock)

- Setup: `uv sync`. **gotcha:** `ruff` and `mypy` live in the `dev` optional-dependency, so plain `uv sync` does NOT put them on PATH (`uv run ruff` fails). Run `uv sync --extra dev` (or `uv pip install ruff mypy`, as CI does).
- Test: `uv run pytest -q` (41 tests pass). Single: `uv run pytest tests/unit/test_data_loader.py -q`.
- Lint: `uv run ruff check .` — pyproject: line-length 100, `E501` ignored, selects `E,F,I,W,UP,B`; extra per-file ignores for `scripts/*.py`.
- Typecheck: `uv run mypy src` — **non-blocking** in CI; only a small subset is checked (`config/`, `evaluation/`, `experiments/`, `utils/`, `models/fusion.py`; most of `models/` and tests are excluded).
- Run: `uv run msdl-jci --mode {proposed,ablation,legacy} --epochs N`, or `uv run msdl-jci-ablation`. **No Makefile exists** despite README's `make` mention. README fallback `pip install -e ".[dev]"` also works.
- To execute the notebook: `uv run jupyter execute notebooks/01_data_alignment_example.ipynb --inplace` (needs `uv pip install nbformat nbconvert`; already verified clean — 6 code cells, all pass).
- Settings load optional `.env` / `development/.env` but have working defaults (`src/msdl_jci/config/settings.py`).

## Data (what is actually in git)

- The 4 raw CSVs in `data/raw/` ARE tracked in git even though `.gitignore` lists `data/raw/*` (tracked before the ignore rule). New files under `data/raw/` are ignored.
- `data/processed/daily_news_embeddings.csv` (768-d IndoBERT vectors) is git-ignored and presently absent; when missing, `dataset_builder.py` silently substitutes all-zero news vectors, so the news modality has no signal locally or in CI.
- Tests are fully synthetic (generated in-code). `tests/conftest.py` adds `src/` to `sys.path`. There is no `data/sample/aligned_sample.csv` (README is stale here).
- `scripts/` holds the notebook generator (`01_build_data_alignment_notebook.py`); its `README.md` table of diagnostic scripts is stale (those scripts were removed). Regression outputs in `data/processed/` are untracked/ignored; artifacts under `artifacts/` are git-ignored, `reports/` is committed.

## Leak-free methodology — do not "fix" these

- Order: align → **train-only scaling** → window (lookback 28) → embargoed walk-forward (70% train, `embargo=5` = label horizon) → AdamW + early stop → validation-threshold → frozen test eval.
- Always pass `train_end_idx` when fitting scalers; fitting on the full frame leaks test ranges (builder warns and `evaluation/walk_forward.py` splits accordingly).
- Macro alignment is ffill-only with leading NaNs dropped — never backfill (`bfill` injects future macro values).
- Target = `future_close > Close` at t+5. Seeds are 42 throughout (`set_all_seeds`).
- `main()` auto-selects `legacy` mode when `sys.argv` contains `pytest` (main.py:93) so the integration test stays fast.
- Concrete Table 1 (alignment matrix) is demonstrated in `notebooks/01_data_alignment_example.ipynb` — a runnable notebook over the real `data/raw/` CSVs showing causal technical indicators, point-in-time ffill macro alignment, t+5 labeling, and the zero-vector news fallback (`dataset_builder.py`). A generator script lives at `scripts/01_build_data_alignment_notebook.py`.