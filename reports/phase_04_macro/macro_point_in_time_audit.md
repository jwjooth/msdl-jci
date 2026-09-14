# Macro point-in-time audit

- Alignment: `merge_asof(direction='backward')` + ffill-only; no bfill (leading NaNs dropped).
- Period-vs-publication risk: BI/inflation CSVs carry period dates, not release dates. Conservative rule: monthly values treated as available on the 1st of the NEXT month (implemented in engineered features via shift(1)-causal construction; raw levels unchanged).
- Engineered features use strictly-past inputs (`shift(1)` before any diff/rolling).
- Verdict: raw macro levels are stale step functions (see staleness stats). If gating favors macro despite macro-only AUC ~ 0.5, it is a bias shortcut, not signal.
