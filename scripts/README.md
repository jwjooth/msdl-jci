# Scripts

Diagnostic and audit scripts. All read the frozen pipeline; none change conclusions.

| Script | Purpose | Output |
|---|---|---|
| `run_full_audit.py` | Leak-free baselines + neural models | `reports/` comparison CSV/MD |
| `diagnose_evaluation.py` | Prob histograms, ROC, calibration | `reports/` diagnostics |
| `diagnose_collapse.py` | Phase 1 collapse probe | `reports/phase_01_collapse/` |
| `diagnose_branches.py` | Phase 2 modality probes | `reports/phase_02_branch_diagnosis/` |
| `diagnose_gating.py` | Phase 3 gating experiments | `reports/phase_03_gating/` |
| `diagnose_macro.py` | Phase 4 macro audit | `reports/phase_04_macro/` |
| `diagnose_news.py` | Phase 5 news audit | `reports/phase_05_news/` |
| `sweep_training.py` | Phase 6 dynamics sweeps | `reports/phase_06_training_dynamics/` |
| `compare_objectives.py` | Phase 7 loss comparison | `reports/phase_07_objective_threshold/` |
| `robust_validation.py` | Phase 9 folds × seeds | `reports/phase_09_robust_validation/` |
| `repair_trading.py` | Phase 10 trading plots | `reports/phase_10_trading/` |
| `capture_phase00_baseline.py` | Phase 0 reproduction capture | `reports/phase_00_baseline/` |

Run with `PYTHONPATH=src python3 scripts/<name>.py` (or `uv run python scripts/<name>.py`).
