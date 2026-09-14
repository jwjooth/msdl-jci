# Objective recommendation (VAL-selected)

- bce+pos_weight: val_auc=0.4408 val_mcc=-0.009 val_p_std=0.00329 val_thr=0.2
- bce-plain: val_auc=0.4508 val_mcc=0.0 val_p_std=0.00525 val_thr=0.5500000000000003
- focal-g2: val_auc=0.4718 val_mcc=0.0 val_p_std=0.00262 val_thr=0.2
- smooth-0.05: val_auc=0.4552 val_mcc=0.0511 val_p_std=0.0037 val_thr=0.49000000000000027

No objective lifts VAL AUC above ~0.53 or VAL p_std out of the near-constant band; keep BCE+pos_weight (simplest, calibrated) and do NOT chase objectives to fix a signal problem. Threshold: validation-MCC.
