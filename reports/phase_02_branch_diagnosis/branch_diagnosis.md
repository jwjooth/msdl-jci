# Branch diagnosis

- Representation stats: see branch_stats.json.
- Branch-only logreg test AUC: tech=0.4899, macro=0.505, news=0.5017.
- Branch-only MCC: tech=0.0108, macro=0.0, news=0.0329.
- One-batch grad norms: {'tech': 0.06691, 'macro': 0.03163, 'news': 0.18779, 'gating': 0.05407, 'classifier': 0.18094}.
- Gate mean at init: [0.329, 0.347, 0.324] (expect ~0.333 each).

Interpretation: macro raw features are low-frequency/stale — if macro-only AUC ~ 0.5 yet the fused gate collapses to macro (beta→0.9), the gate is exploiting macro as a bias shortcut, not signal. News-only signal is typically weak (high-dim, sparse). Tech carries most genuine signal.
