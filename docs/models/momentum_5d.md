# Model card — Momentum-5d

```text
Status: Research heuristic.
Reason: Strongest observed baseline, but requires further validation.
```

- Rule: P(UP) from 28-day scaled close change; no training.
- Observed (single test slice): net ≈ 28.07, Sharpe ≈ 1.31, exposure ≈ 55%.
- Must be presented as heuristic needing out-of-sample validation, never as DL success.
