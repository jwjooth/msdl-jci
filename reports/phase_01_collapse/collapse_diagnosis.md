# Collapse diagnosis (5-epoch probe, seed 42)

- val p_std=0.00112, test p_std=0.00301
- test logit_std=0.01206
- val_thr(MCC)=0.20
- overfit_32: {'init_loss': 0.6722443103790283, 'final_loss': 0.6351677179336548, 'acc': 0.625, 'grad_first': 0.3626570701599121, 'grad_last': 2.2426083087921143, 'pass': False}
- verdict: **COLLAPSE CONFIRMED**

Root-cause hypothesis: gating saturates toward the low-variance macro branch (see Phase 3); near-constant logits ⇒ sigmoid compresses all mass into a ~0.01 band. Threshold tuning only slides the decision point inside noise — MCC stays ~0. Fix gating/branch signal before any hyperparameter tuning.
