# 08 — Reproducibility

## Preferred (uv)
```bash
uv sync
uv run pytest -v
uv run msdl-jci --mode proposed --epochs 40
uv run python -m msdl_jci.experiments.run_ablation
```

## Fallback (no uv)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -v
python -m msdl_jci.main --mode proposed --epochs 40
python -m msdl_jci.experiments.run_ablation
```

## Environment notes
- Some audit runs used `PYTHONPATH=src python3 ...` because `uv` was unavailable
  in a Linux shell over a Windows-built `.venv`. Same code and seed (42); only
  the interpreter differed. Recorded in `reports/phase_00_baseline/`.
- Windows: the committed `.venv/` is **not** portable — recreate it locally
  (git-ignored). Use `make` targets (`install lint format type test smoke`).
- Seeds: `set_all_seeds()` pins Python/NumPy/Torch/CUDA/dataloader; cuDNN deterministic.
- Configs: `configs/default.toml` is the frozen thesis setup; overrides in
  `configs/{models,experiments,evaluation,trading}/`.
- Lint: `ruff check .`; types: `mypy src` (missing-imports ignored; documented
  exceptions minimal). CI: `.github/workflows/ci.yml`.
