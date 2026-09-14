# Reproducibility Snapshot (Phase 0)

- Date: 2026-09-14. Seeds: `seed=42` everywhere via `set_all_seeds()`
  (Python `random`, NumPy, Torch CPU/CUDA, DataLoader generator, cuDNN deterministic).
- torch 2.14.0+cpu (this shell), sklearn/pandas/numpy per `uv.lock` (Windows `.venv` for `uv` runs).
- Config: `ML_LOOK_BACK=28`, `ML_PREDICTION_HORIZON=5`, `ML_RANDOM_STATE=42`,
  `LSTM_HIDDEN_DIM=64`, `MACRO (3→32→16)`, `NEWS 768→64`, `FUSION 144→32→1`.
- Split: single walk-forward 70/10/20 with `embargo=5` gaps
  (train `[:train_end]`, val `[train_end+5:val_end]`, test `[val_end+5:]`).
- Scalers: `MinMaxScaler` fit on train-prefix df rows only
  (`train_end_df = train_end_win + lookback - 1`), transform all.
- `pos_weight` = train neg/pos, BCEWithLogitsLoss; AdamW lr=1e-3, wd=1e-4,
  ReduceLROnPlateau(factor 0.5, patience 4), grad-clip 1.0, early stop patience 10 on val loss.
- Threshold: validation-only MCC grid [0.20, 0.80] step 0.01 (`find_best_threshold`).
- Costs: fee 0.0015/side + slippage 0.0005/side; stride 5; long/flat; cash earns
  risk-free 5% annualized; benchmark = buy-and-hold on same stride dates.
- Note: `uv` unavailable in this Linux shell (Windows `.venv`); commands run as
  `PYTHONPATH=src python3 -m pytest` / `PYTHONPATH=src python3 scripts/...`.
  Equivalence: same code, same seed; only interpreter differs.
