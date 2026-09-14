# 02 — Architecture

## Module map (current `src/msdl_jci/` layout; stable API, tests pin it)

| Concept | Module |
|---|---|
| Settings/paths | `config/settings.py` |
| Dataset builder, indicators, alignment | `utils/dataset_builder.py` |
| Causal macro features | `utils/macro_features.py` |
| Branches, gating, fusion, baselines | `models/fusion.py` |
| Walk-forward, seeding, thresholds, losses | `evaluation/walk_forward.py` |
| Robust multi-fold validation | `evaluation/robust_walk_forward.py` |
| Metrics (incl. MCC, PR-AUC, Brier) | `evaluation/metrics.py` |
| Trading simulator (costs, Sortino, exposure) | `evaluation/trading_simulation.py` |
| Ablation runner / entrypoints | `experiments/run_ablation.py`, `main.py` |

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
