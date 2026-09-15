# 02 — Architecture

## Module map (current `src/msdl_jci/` layout; stable API, tests pin it)

| Concept | Module |
|---|---|
| Settings/paths | `src/msdl_jci/config/settings.py` |
| Dataset builder, indicators, alignment | `src/msdl_jci/utils/dataset_builder.py` |
| Causal macro features | `src/msdl_jci/utils/macro_features.py` |
| Model branches (LSTM, MLP, News), gating, fusion | `src/msdl_jci/models/fusion.py` |
| Walk-forward validation, seeding, thresholds, losses | `src/msdl_jci/evaluation/walk_forward.py` |
| Metrics (MCC, PR-AUC, Brier, etc.) | `src/msdl_jci/evaluation/metrics.py` |
| Trading simulator (costs, Sortino, exposure) | `src/msdl_jci/evaluation/trading_simulation.py` |
| Ablation runner / CLI entrypoints | `src/msdl_jci/experiments/run_ablation.py`, `src/msdl_jci/main.py` |

## Structure changes
- Removed legacy `src/msdl_jci/{core,domain,infrastructure,services,scraping,preprocessing}/`.
- Removed duplicate `models/`, `utils/`, `config/` directories at repo root.
- Removed diagnostic scripts in `scripts/` (replaced by `reports/`).
- Added `notebooks/thesis_defense.ipynb` for defense visualization.

## Model branches
- Technical LSTM (7 → 64, 2 layers, LayerNorm, dropout 0.2, lookback 28).
- Macro MLP (3 → 32 → 16, LayerNorm — BatchNorm removed for small-batch stability).
- News projection (768 → 64, LayerNorm).

## Fusion design
- Gating: 144 → 64 → 3 softmax (near-zero init → uniform); options:
  `temperature`, `min_weight` floor, entropy bonus, modality dropout, aux heads.
- Classifier: 144 → 32 → 1 with `pos_rate` bias init.
- Baselines: PureLSTM, LSTM+Macro, LSTM+News, StaticFusion + heuristics/ML.

## Flows
Training: align → train-only scale → window → embargo split → AdamW + early stop
(val loss/MCC/AUC) → val MCC threshold → frozen test eval.
Evaluation: classification metrics + trading sim → per-phase `reports/`.
