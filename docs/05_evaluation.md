# 05 — Evaluation

- **Split.** Single 70/10/20 + **embargo 5** (horizon t+5); robust: 2 expanding
  folds × 3 seeds (42, 43, 44).
- **Selection on validation only.** Early stopping (loss/MCC/AUC), MCC threshold
  grid [0.20, 0.80], pos_weight from train, architecture choices.
- **Metrics.** Accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC,
  PR-AUC, log loss, Brier, pred-pos-rate, prob-std, confusion matrix, calibration.
- **Collapse rule.** p_std < 0.01 + MCC ≈ 0 ⇒ collapse (warned in logs).
- **Statistics.** Wilcoxon + paired-t on fold AUC; bootstrap CI on return gaps.
- **Result.** No pairwise model difference significant; see
  `reports/phase_09_robust_validation/statistical_tests.md`.
