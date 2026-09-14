# Model card — Adaptive Soft Gating Fusion (proposed)

```text
Status: Not viable.
Reason: Model collapse, weak branch signal, no robust improvement over baselines.
```

- Inputs: tech windows [28,7], macro [3], news [768]. Output: t+5 UP probability.
- Training: leak-free WF + embargo, BCE+posweight, val-MCC threshold.
- Metrics: AUC ≈ 0.53, MCC ≈ 0.02, p_std ≈ 0.001; net 3.50 vs benchmark 25.32.
- Limits: collapses; aux losses harm; stabilizers balance gates only.
- Risk: must not be traded or presented as working. Maintenance: frozen.
