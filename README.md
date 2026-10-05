# MSDL-JCI — Multi-Source Deep Learning for JCI Direction Prediction

Predicting Indonesia Composite Index (t+5) direction from technical, macroeconomic, and news modalities — implements the **thesis defense architecture**.

```text
Research Status: Thesis-aligned implementation, honestly trained — walk-forward pooled AUC ≈ 0.51 (per-fold 0.48–0.62), tech-dominated gating.
Proposed Model: LSTM + MLP Encoder + Frozen IndoBERT + Adaptive MLP Fusion (Soft Gating).
Deployment Status: Research/thesis defense only. Not for live trading.
```

This repo is a **single Jupyter notebook**: `notebooks/development.ipynb` contains the config, hyperparameters (Table 5), SQLite entity contract, all utilities, the model architecture, walk-forward evaluation, visualizations, and leak-free checks.

## Architecture (Thesis Table 5)

| Component | Specification |
|---|---|
| Frozen IndoBERT | `indobenchmark/indobert-base-p1` (frozen CLS, precomputed + cached) |
| Projection | Linear (768 → 64) |
| LSTM Branch | Hidden=64, Lookback=28, Dropout=0.2 |
| MLP Encoder | 2 layers (3→32→16), ReLU |
| Adaptive Fusion | Linear(144→3) + Softmax → (α, β, γ) |
| Head | Linear(64→1), BCEWithLogitsLoss |

## Repository structure

```text
MSDL-JCI/
├─ notebooks/development.ipynb   # the whole project (config → utils → model → run → viz → checks)
├─ database/main_database.db     # SQLite source of truth (git-ignored, not shipped)
├─ pyproject.toml uv.lock .python-version .env.example
├─ AGENTS.md README.md
```

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env  # optional: defaults work without it
uv run python -m ipykernel install --user --name python3
uv run jupyter execute notebooks/development.ipynb --inplace
```

`uv.lock` is the single source of truth.

## Data requirements

Single SQLite file (git-ignored, `DATABASE_PATH` env overrides):

```text
database/main_database.db   # jci_historical, bi_rate, inflation_data, kurs_usdidr
                             # + cnbc/detik/kontan_ihsg_articles
```

Absent DB → the notebook runs a synthetic seed-42 fallback instead. News vectors come from detik articles (2018–2025) embedded once with frozen IndoBERT and cached to `database/news_emb_cache.npz`, attached past-only; CNBC/Kontan articles are dated 2026 (past JCI end) and excluded.

## Evaluation methodology

- Walk-forward expanding window validation (5 splits)
- Metrics: AUC, Accuracy, F1-Score (binary classification)
- Per-fold metrics, gating-weight (α, β, γ) evolution/distribution, ROC, confusion matrices, prediction-vs-actual, confidence plots

## Limitations and risks

Not investment advice; not live-trading ready; historical simulation only. Cost assumptions may change results; macro release-date and news-timing assumptions are imperfect; small, non-stationary sample.

## License

MIT.