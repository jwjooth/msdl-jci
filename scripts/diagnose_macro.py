"""Macro audit (Phase 4): coverage, staleness, point-in-time safety.

Saves to reports/phase_04_macro/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_04_macro"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from msdl_jci.config.settings import get_settings
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    s = get_settings()
    b = MultiSourceDatasetBuilder()
    df_bi, df_inf, df_kurs = b.load_raw_macro_data(s.BI_RATE_CSV, s.INFLATION_CSV, s.KURS_CSV)
    df = b.build_aligned_dataframe()

    cov = {"n_macro_obs": {"bi": len(df_bi), "inflation": len(df_inf), "kurs": len(df_kurs)},
           "n_daily_samples": len(df),
           "date_range": [str(df["Date"].min()), str(df["Date"].max())]}
    (OUT / "macro_coverage.json").write_text(json.dumps(cov, indent=2))

    # Daily variance: how stale is each macro series after alignment?
    rows = []
    for c in ("bi_rate", "inflation_rate", "usd_idr"):
        v = df[c]
        rows.append({"col": c, "n_unique": int(v.nunique()), "n_rows": len(v),
                     "frac_unique": round(float(v.nunique()) / len(v), 4),
                     "daily_change_std": float(v.diff().std()),
                     "frac_days_changed": round(float((v.diff().fillna(0) != 0).mean()), 4)})
    pd.DataFrame(rows).to_csv(OUT / "macro_daily_variance.csv", index=False)

    # Staleness: run lengths of constant stretches.
    stale = {}
    for c in ("bi_rate", "inflation_rate", "usd_idr"):
        runs, cur = [], 1
        vals = df[c].values
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        stale[c] = {"median_stale_days": float(np.median(runs)),
                    "max_stale_days": int(np.max(runs)),
                    "mean_stale_days": float(np.mean(runs))}
    (OUT / "macro_staleness_stats.json").write_text(json.dumps(stale, indent=2))

    (OUT / "macro_feature_proposals.md").write_text(
        "# Macro feature proposals (causal; see utils/macro_features.py)\n\n"
        "- Change features: `{bi_rate,inflation_rate,usd_idr}_chg_{21,63,126}d` on shift(1) values.\n"
        "- Z-scores: `usd_idr_z{126,252}d` vs rolling past-only mean/std.\n"
        "- Regimes: `rate_hiking/cutting`, `high_inflation`, `idr_depreciating`.\n"
        "- Capacity: shrink macro MLP (e.g. 3->16->8), dropout 0.2, early stopping;\n"
        "  stale levels alone cannot justify gating dominance.\n")

    (OUT / "macro_point_in_time_audit.md").write_text(
        "# Macro point-in-time audit\n\n"
        "- Alignment: `merge_asof(direction='backward')` + ffill-only; no bfill (leading NaNs dropped).\n"
        "- Period-vs-publication risk: BI/inflation CSVs carry period dates, not release dates. "
        "Conservative rule: monthly values treated as available on the 1st of the NEXT month "
        "(implemented in engineered features via shift(1)-causal construction; raw levels unchanged).\n"
        "- Engineered features use strictly-past inputs (`shift(1)` before any diff/rolling).\n"
        "- Verdict: raw macro levels are stale step functions (see staleness stats). "
        "If gating favors macro despite macro-only AUC ~ 0.5, it is a bias shortcut, not signal.\n")
    print(json.dumps({"coverage": cov, "staleness": stale}, indent=2))


if __name__ == "__main__":
    main()
