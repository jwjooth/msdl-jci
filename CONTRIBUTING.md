# Contributing

1. Do not alter the accepted research conclusions without new evidence reviewed
   by a maintainer (see `reports/final_recommendation.md`).
2. Preserve leakage safeguards: train-only scaling, embargo, point-in-time
   macro handling, news date alignment, validation-only thresholds.
3. Add/extend tests with every behavior change; run `pytest -v`.
4. Run `ruff check .` and `ruff format .` before pushing.
5. Never commit secrets, raw data, checkpoints, or large artifacts.
6. Document environment: `uv` preferred; otherwise note the fallback used.
