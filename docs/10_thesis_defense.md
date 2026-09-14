# 10 — Thesis Defense

## Narrative (5 minutes)
1. **Question.** Can technical + macro + news modalities predict JCI t+5 direction?
2. **Method.** Leak-free walk-forward system with embargo, cost-aware trading.
3. **Finding.** No — the proposed fusion collapses; branches show AUC ≈ 0.50;
   gating fixes balance weights but not signal; nothing beats baselines robustly.
4. **Contribution.** Methodology + diagnosis rigor: train-only scaling, embargo,
   MCC/threshold discipline, gross-vs-net honesty, multi-seed statistics.
5. **Close.** Negative result with full evidence; LSTM+Macro retained as baseline
   only; future work = better-timed data, target redesign.

## Anticipated questions
- *Why not tune more?* Sweeps (lr/wd/dropout/objectives/gating) all stayed ~random
  on validation — tuning noise is not signal (`phase_06/07/03`).
- *Is LSTM+Macro deployable?* No — n.s. vs proposed, often ≈ buy-and-hold.
- *Momentum-5d?* Heuristic needing validation, not a DL claim.
- *Leakage?* Enumerate 8 safeguards + tests (`test_no_lookahead.py`, `test_target_alignment.py`).

## Pointers
Executive summary: `docs/01_executive_summary.md`. Verdict: `reports/final_recommendation.md`.
Slides: use key tables from `phase_08/09/10` (gross AND net, exposure noted).
