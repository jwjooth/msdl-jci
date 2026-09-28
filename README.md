# MSDL-JCI — Multi-Source Deep Learning for JCI Direction Prediction

Predicting Indonesia Composite Index (t+5) direction from technical, macroeconomic, and news modalities — closed as a rigorous, leak-free **negative result**.

```text
Research Status: Closed.
Proposed Adaptive Soft Gating: Not viable.
Best research baseline: LSTM + Macro.
Strongest simple heuristic: Momentum-5d.
Deployment Status: Research/thesis defense only. Not for live trading.
```

## Executive summary

- **Built:** leak-free multimodal pipeline (point-in-time alignment, train-only scaling, embargoed walk-forward, validation-only thresholds, cost-aware trading sim).
- **Tested:** 5 neural models + 9 statistical/ML baselines, 2 folds × 3 seeds, Wilcoxon/t-test + bootstrap statistics.
- **Failed:** the proposed Adaptive Soft Gating model collapsed (near-constant probabilities, MCC ≈ 0); branch probes showed AUC ≈ 0.50 per modality; gating fixes balanced weights but not signal.
- **Learned:** the bottleneck is data signal (stale macro, sparse news, weak technical edge at t+5), not engineering.
- **Useful:** evaluation methodology, diagnostic suite, and honest negative evidence. LSTM + Macro retained as research baseline only; nothing is deployable.

> Branch note (`refactor`): simplification branch — data lives in git-ignored SQLite (`database/main_database.db`), not CSVs. Suite is green (41 passed); the only runnable artifact besides the package is `notebooks/development.ipynb` (self-contained synthetic demo).

## Key results

| Item | Finding | Evidence |
|---|---:|---|
| Proposed model | Not viable | MCC near zero, probability collapse |
| Branch-only AUC | ~0.50 | branch probes |
| Gating fixes | Stabilized weights but not signal | entropy improved, MCC unchanged |
| LSTM + Macro | Research baseline only | mean AUC ~0.588, not significant |
| Momentum-5d | Strongest heuristic | net ~28.07, Sharpe ~1.31, exposure ~55% |

Statistical significance is **not** claimed where tests were null (all p > 0.05).

## Architecture

```mermaid
flowchart TD
  A[Raw Technical Data] --> B[Technical Features]
  C[Raw Macro Data] --> D[Point-in-Time Macro Features]
  E[News Embeddings] --> F[News Alignment Features]
  B --> G[Windowed Tensors]
  D --> G
  F --> G
  G --> H[Walk-Forward Split + Embargo]
  H --> I[Train-Only Scaling]
  I --> J[Model Training]
  J --> K[Validation Threshold Optimization]
  K --> L[Test Evaluation]
  L --> M[Classification Metrics]
  L --> N[Trading Simulation with Costs]
```

## Repository structure

```text
MSDL-JCI/
├─ .github/workflows/ci.yml    # CI pipeline (uv sync, advisory ruff, non-blocking mypy, pytest)
├─ notebooks/                  # development.ipynb (runnable synthetic demo)
├─ src/msdl_jci/               # package (config, models, evaluation, experiments, utils)
├─ tests/                      # unit / integration (synthetic, no data files)
├─ pyproject.toml uv.lock .python-version
├─ AGENTS.md README.md
```

Shipped without: raw/processed data, `docs/`, `reports/`, `scripts/`, `configs/`, `Makefile`, `Dockerfile`. See AGENTS.md for what is actually on disk.

## Quickstart

Pytest creates a temporary SQLite database with synthetic data automatically;
no local database is needed to run tests.

Before running either training command below, obtain a populated copy of
`main_database.db` from the project maintainer and place it at
`database/main_database.db` (create the `database/` directory first). Alternatively,
set `DATABASE_PATH` in your shell or `.env` to an existing populated SQLite file.
The database is git-ignored and is not included in a fresh checkout; this branch
has no data download or import script. Required table columns are listed in
`ENTITY_TABLES` in `src/msdl_jci/utils/data_loader.py`. An empty SQLite file is
not sufficient.

```bash
uv sync --extra dev
cp .env.example .env  # optional: defaults work without it
uv run pytest -q
uv run msdl-jci --mode proposed --epochs 40
uv run msdl-jci-ablation
```

Fallback (no uv):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pytest -q
python -m msdl_jci.main --mode proposed --epochs 40
```

`uv.lock` is the single source of truth for dependencies. Lint (`ruff`) and typecheck (`mypy`) come from the `dev` extra, so plain `uv sync` won't put them on PATH.

## Data requirements

Single SQLite file (git-ignored, `DATABASE_PATH` env overrides):

```text
database/main_database.db   # jci_historical, bi_rate, inflation_data, kurs_usdidr
                            # + cnbc/detik/kontan_ihsg_articles
```

Table shapes match the old CSVs, so explicit CSV paths still work as overrides everywhere. The DB has article tables but no embedding vectors — when no embedding CSV is given, the builder substitutes all-zero news vectors, so the news modality carries no signal. Tests use a temporary SQLite DB seeded with synthetic rows (encoder/builder/pipeline) or synthetic DataFrames generated in-code; they do not read the contributor’s database.

## Dev notebook

`notebooks/development.ipynb` — self-contained runnable demo of the builder logic on synthetic data: causal technical indicators, point-in-time ffill macro alignment, t+5 labeling, train-only scaler fitting (`train_end_idx`), and windowed tensor construction. Execute with:

```bash
uv run jupyter execute notebooks/development.ipynb --inplace
```

It is the only notebook; the stale `01_*`/`app.ipynb` were deleted.

## Limitations and risks

Not investment advice; not live-trading ready; historical simulation only. Cost assumptions may change results; macro release-date and news-timing assumptions are imperfect; small, non-stationary sample.

## License

MIT.
