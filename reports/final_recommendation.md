# Final Recommendation — MSDL-JCI (thesis-ready)

## 1. Executive Summary
The proposed Adaptive Soft Gating model is **not viable** (collapse, MCC ≈ 0).
LSTM + Macro is a **research baseline only** (mean AUC ≈ 0.588, not significant).
**Nothing is deployable for live trading.** Closed as a rigorous negative result.

## 2. Background
t+5 JCI direction from technical + macro + news modalities; 5 neural models vs
9 statistical/ML baselines under a leak-free protocol.

## 3. Evaluation Protocol
Train-only scaling; embargo 5; 70/10/20 + 2 folds × 3 seeds; val-only thresholds
(MCC), pos_weight (train), early stopping; MCC/AUC/PR-AUC/Brier/calibration;
gross-vs-net trading with fees 0.0015 + slippage 0.0005 per side.

## 4. Diagnosis of Model Collapse
Test p_std 0.001–0.005, logit_std ≈ 0.012; 32-sample overfit fails at 60 steps,
partial at 200 (loss 0.69→0.20, acc 0.875) — pipeline works, signal missing.
Evidence: `phase_01_collapse/`, `phase_06_training_dynamics/overfit_test.md`.

## 5. Branch-Level Evidence
Branch-only logreg AUC ≈ 0.50 (tech 0.49, macro 0.505, news 0.50); macro MCC 0.0.
Gate init balanced (0.329/0.347/0.324) — collapse emerges in training.
Evidence: `phase_02_branch_diagnosis/`.

## 6. Gating Analysis
Temperature/min-weight/entropy fixes raise entropy 0.86→1.09 and balance weights,
but MCC stays ≈ 0; aux losses harm (AUC 0.38). Gating fixed, signal absent.
Evidence: `phase_03_gating/gating_fix_experiments.md`.

## 7. Macro Staleness Finding
BI median stale 33 d (max 367 d); only 12 unique values in 1908 rows. Macro-only
AUC 0.505 — gating-to-macro is a bias shortcut. Causal change features added
(`utils/macro_features.py`) but cannot manufacture signal.
Evidence: `phase_04_macro/`.

## 8. News Feature Finding
7.1% zero-vector days; exact-date merge (no leak); linear/MLP probes AUC 0.50–0.54.
High-dim sparse input justifies the bottleneck, not expansion.
Evidence: `phase_05_news/`.

## 9. Robust Validation Results
4 models × 3 seeds × 2 folds: AUC 0.52–0.59; no pairwise difference significant
(Wilcoxon/t p > 0.05); bootstrap CI for (LSTM+Macro − Proposed) net return
[−3.18, +9.89] includes 0. Evidence: `phase_09_robust_validation/`.

## 10. Trading Simulation Results
Proposed net 3.50 vs gross 9.03 (5.53 pp drag) vs benchmark 25.32. ~100% exposure
rows ≈ buy-and-hold minus costs. Momentum-5d: net 28.07, Sharpe 1.31, exposure
55% — heuristic needing validation. Evidence: `phase_10_trading/`.

## 11. Statistical Interpretation
No effect reaches significance; mean-best ranks are descriptive, not claims.
Threshold sensitivity sweeps show decisions sliding inside noise (MCC ≈ 0).

## 12. Final Verdict
- Proposed Adaptive Soft Gating: **not viable** — do not trade, do not present as working.
- LSTM + Macro: **research baseline only**.
- Momentum-5d: **research heuristic**, further validation required.
- Project: **closed negative result**; contribution is methodology + diagnosis rigor.
- Next: better-timed data, target-horizon study, branch-signal gating precondition
  (standalone AUC > 0.55 before any fusion complexity).
