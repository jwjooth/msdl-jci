# 11 — Deployment (research inference only)

```text
This repository is research-grade and suitable for thesis defense, reproduction, and offline analysis.
It is NOT approved for live financial trading.
Live deployment would require additional risk controls, monitoring, compliance, data quality checks, and independent validation.
```

- Offline inference: load aligned tensors → `evaluate_model_walk_forward` with a
  frozen checkpoint → `simulate_trading_strategy` for hypothetical analysis.
- Docker: `docker build -t msdl-jci .` (research image; freezes deps, not a trading system).
- No serving, scheduling, or broker connectivity is included by design.
