# 01 — Executive Summary

**Problem.** Predict JCI direction at horizon t+5 from technical, macro, and news modalities.

**Method.** Leak-free pipeline: point-in-time alignment → train-only scaling →
embargoed walk-forward (70/10/20, embargo 5) → 5 neural models + 9 statistical/ML
baselines → validation-only thresholding → cost-aware trading simulation.

**Result (closed).** The proposed Adaptive Soft Gating model collapsed
(near-constant probabilities, MCC ≈ 0, AUC ≈ 0.53) and is **not viable**.
Branch-only probes showed AUC ≈ 0.50 per modality; gating stabilization fixed
weight balance but not signal. LSTM + Macro (mean AUC ≈ 0.588, n.s.) is a
research baseline only. The momentum-5d heuristic (net ≈ 28.07, Sharpe ≈ 1.31)
is the strongest simple baseline — requiring further validation, not a DL success.

**Conclusion.** Rigorous negative result: nothing is deployable for live trading.
Contribution: leakage-free evaluation methodology, collapse diagnosis, and honest
reporting. Evidence: `reports/` phases 00–11 and `final_recommendation.md`.
