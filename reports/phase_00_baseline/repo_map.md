# MSDL-JCI Repository Map (Phase 0 — frozen baseline)

## Config
- `src/msdl_jci/config/settings.py` — `Settings` dataclass + `get_settings()` (cached).
  Key values: `ML_LOOK_BACK=28`, `ML_PREDICTION_HORIZON=5`, `ML_RANDOM_STATE=42`,
  `LSTM_HIDDEN_DIM=64`, `NEWS_EMB_DIM=768`, `FUSION_INPUT_DIM=144`.
  Data paths: `data/raw/macro/{bi_rate,inflation_data,kurs_usdidr}.csv`,
  `data/raw/technical/jci_historical.csv`, `data/processed/daily_news_embeddings.csv`.

## Dataset builder
- `src/msdl_jci/utils/dataset_builder.py`
  - `calculate_technical_indicators()` — RSI-14, MACD(12,26,9), ATR-14, SMA-20.
  - `MultiSourceDatasetBuilder.build_aligned_dataframe()` — JCI + macro
    (`merge_asof backward` + **ffill only**, leading NaNs dropped), t+5 labeling
    (`future_close`, `return_5d`, `target_direction`), news left-merge with
    zero-vector imputation.
  - `create_multisource_tensors(..., train_end_idx)` — scalers fit on
    `[:train_end_idx]` rows when given (leak-free); leakage guard rejects
    `future_close`/`return_5d`/`target_direction` in features; sliding windows
    `lb=28` preserving chronology.

## Walk-forward evaluator / training loop
- `src/msdl_jci/evaluation/walk_forward.py`
  - `set_all_seeds()`, `compute_pos_weight()` (train-only neg/pos),
    `find_best_threshold()` (validation MCC grid 0.20–0.80).
  - `train_single_split()` — AdamW + ReduceLROnPlateau + early stopping on val
    loss; per-epoch diagnostics (train/val loss, val AUC/bal-acc, pos ratio,
    p_mean/p_std, grad norm, gating means); collapse warnings; val-threshold
    (MCC) + test inference. Returns `(model, probs, weights, best_thr, history)`.
  - `evaluate_model_walk_forward()` — 70/10/20 split with `embargo=5` gaps;
    threshold override or val-optimized; test metrics + trading sim.
  - `MultiSourceTorchDataset`, `EvaluationResults` (+`best_threshold`, `history`).

## Model definitions
- `src/msdl_jci/models/fusion.py`
  - `TechnicalLSTMBranch` (7→64, 2-layer LSTM, LayerNorm, dropout 0.2).
  - `MacroMLPBranch` (3→32→16, LayerNorm — BatchNorm removed).
  - `NewsProjectionBranch` (768→64, LayerNorm).
  - `SoftGatingNetwork` (144→64→3, near-zero init → ~uniform).
  - `AdaptiveSoftGatingFusionModel` (+`pos_rate` bias init, per-branch
    LayerNorm, `modality_dropout`).
  - Baselines: `PureLSTMModel`, `LSTMMacroModel`, `LSTMNewsModel`,
    `StaticFusionModel`.
- `src/msdl_jci/evaluation/robust_walk_forward.py` — expanding-window
  multi-fold WF (`expanding_window_splits`, `run_robust_walk_forward`).

## Threshold optimizer
- `find_best_threshold()` in `walk_forward.py` — **validation only**.

## Trading simulator
- `src/msdl_jci/evaluation/trading_simulation.py` — `simulate_trading_strategy()`
  long/flat on 5-day stride; costs (`transaction_cost=0.0015`,
  `slippage=0.0005` per side); gross vs net, Sharpe/Sortino, MDD, win rate,
  exposure, turnover.

## Ablation runner
- `src/msdl_jci/experiments/run_ablation.py` — leak-free tensors
  (train-prefix scalers), 5 models, `seed=42` passed through, CSVs + plots
  to `data/processed/`.
- `src/msdl_jci/main.py` — `run_proposed_deep_learning_pipeline(epochs, seed)`
  with seeding + leak-free tensors + calibrated `pos_rate`.

## Metrics
- `src/msdl_jci/evaluation/metrics.py` — accuracy/precision/recall/F1/AUC +
  balanced-acc, MCC, PR-AUC, log-loss, Brier, threshold, pred-pos-rate.

## Audit scripts
- `scripts/run_full_audit.py`, `scripts/diagnose_evaluation.py`.

## Tests (30 passing)
- `tests/unit/test_target_alignment.py`, `test_no_lookahead.py` (leak safeguards),
  plus config/data_loader/dataset_builder/fusion/macro/technical encoders,
  trading_simulation, `tests/integration/test_pipeline.py`.
