# MSDL-JCI — Multi-Source Deep Learning for JCI Direction Prediction

Predicting Indonesia Composite Index (t+5) direction from technical, macroeconomic, and news modalities — closed as a rigorous, leak-free **negative result**.

```text
Research Status: Closed.
Proposed Adaptive Soft Gating: Not viable.
Best research baseline: LSTM + Macro.
Strongest simple heuristic: Momentum-5d.
Deployment Status: Research/thesis defense only. Not for live trading.
```

This repo is a **single Jupyter notebook**: `notebooks/development.ipynb` contains the config, hyperparameters, SQLite entity contract, all utilities, the run over the real database, visualizations, and leak-free checks. No training — closed result.

## Key results

| Item | Finding | Evidence |
|---|---:|---|
| Proposed model | Not viable | MCC near zero, probability collapse |
| Branch-only AUC | ~0.50 | branch probes |
| Gating fixes | Stabilized weights but not signal | entropy improved, MCC unchanged |
| LSTM + Macro | Research baseline only | mean AUC ~0.588, not significant |
| Momentum-5d | Strongest heuristic | net ~28.07, Sharpe ~1.31, exposure ~55% |

Statistical significance is **not** claimed where tests were null (all p > 0.05).

## Repository structure

```text
MSDL-JCI/
├─ notebooks/development.ipynb   # the whole project (config → utils → run → viz → checks)
├─ database/main_database.db     # SQLite source of truth (git-ignored, not shipped)
├─ pyproject.toml uv.lock .python-version .env.example
├─ AGENTS.md README.md
```

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env  # optional: defaults work without it
uv run jupyter execute notebooks/development.ipynb --inplace
```

`uv.lock` is the single source of truth. First run needs a `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`.

## Data requirements

Single SQLite file (git-ignored, `DATABASE_PATH` env overrides):

```text
database/main_database.db   # jci_historical, bi_rate, inflation_data, kurs_usdidr
                            # + cnbc/detik/kontan_ihsg_articles
```

Absent DB → the notebook runs a synthetic seed-42 fallback instead. The DB has article tables but no embedding vectors, so the news modality is all-zero vectors with no signal.

## Limitations and risks

Not investment advice; not live-trading ready; historical simulation only. Cost assumptions may change results; macro release-date and news-timing assumptions are imperfect; small, non-stationary sample.

## License

MIT.
