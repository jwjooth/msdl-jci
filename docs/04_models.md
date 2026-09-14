# 04 — Models

| Model | Status | Test AUC | Note |
|---|---|---:|---|
| Adaptive Soft Gating (proposed) | **Not viable** | ~0.53 | Collapse, MCC ≈ 0 |
| Pure LSTM | Baseline | ~0.56–0.59 | Weak, collapses at short epochs |
| LSTM + Macro | Research baseline only | ~0.59 (n.s.) | Best mean, not significant |
| LSTM + News | Baseline | ~0.52–0.54 | Only net with spread (MCC 0.14) |
| Static Fusion | Unreliable | <0.5 | AUC below chance |
| Momentum-5d | Research heuristic | — | net ~28.07, Sharpe ~1.31 |

Details per model: `docs/models/`. Stabilizers (temperature, min-weight,
entropy bonus, modality dropout, aux heads) are implemented but did not rescue
performance — documented as engineering contribution, not a fix.
