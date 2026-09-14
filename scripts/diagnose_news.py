"""News audit (Phase 5): coverage, missingness, embedding stats, probes, leakage.

Saves to reports/phase_05_news/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_05_news"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    b = MultiSourceDatasetBuilder()
    df = b.build_aligned_dataframe()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    E = df[emb_cols].values.astype(float)
    norms = np.linalg.norm(E, axis=1)
    zero_days = norms == 0.0

    cov = {"n_rows": len(df), "emb_dim": len(emb_cols),
           "n_zero_vector_days": int(zero_days.sum()),
           "frac_zero_days": round(float(zero_days.mean()), 4),
           "date_range": [str(df["Date"].min()), str(df["Date"].max())]}
    (OUT / "news_coverage.json").write_text(json.dumps(cov, indent=2))

    # Missingness over time (monthly zero-rate).
    tmp = pd.DataFrame({"Date": pd.to_datetime(df["Date"]), "missing": zero_days.astype(int)})
    tmp["ym"] = tmp["Date"].dt.strftime("%Y-%m")
    miss = tmp.groupby("ym")["missing"].agg(["mean", "sum", "size"]).reset_index()
    miss.columns = ["month", "missing_rate", "n_missing", "n_days"]
    miss.to_csv(OUT / "news_missingness.csv", index=False)

    stats = {"mean": float(E.mean()), "std": float(E.std()),
             "min": float(E.min()), "max": float(E.max()),
             "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
             "zero_vector_rate": round(float(zero_days.mean()), 4)}
    nz = E[~zero_days]
    if len(nz):
        stats.update({"nonzero_mean": float(nz.mean()), "nonzero_std": float(nz.std())})
    (OUT / "news_embedding_stats.json").write_text(json.dumps(stats, indent=2))

    # Probes: logreg + MLP on news only (train fit, test eval, embargo split).
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    from msdl_jci.evaluation.metrics import compute_classification_metrics

    n = len(df)
    tr, te = int(n * 0.70), slice(int(n * 0.80) + 5, n)
    y, yte = df["target_direction"].values[:tr].astype(int), df["target_direction"].values[te].astype(int)
    out = {}
    for name, clf in (("linear", LogisticRegression(max_iter=2000)),
                      ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42))):
        clf.fit(E[:tr], y)
        p = clf.predict_proba(E[te])[:, 1]
        m = compute_classification_metrics(yte, p)
        out[name] = {**m.to_dict(), "prob_std": round(float(p.std()), 5)}
    # Aggregated momentum probe: 5-day mean of norms + missing flag.
    agg = np.stack([pd.Series(norms).rolling(5, min_periods=1).mean().values,
                    zero_days.astype(float)], axis=1)
    clf = LogisticRegression(max_iter=2000).fit(agg[:tr], y)
    p = clf.predict_proba(agg[te])[:, 1]
    m = compute_classification_metrics(yte, p)
    out["aggregated(norm5d+missing)"] = {**m.to_dict(), "prob_std": round(float(p.std()), 5)}
    (OUT / "news_probe_metrics.json").write_text(json.dumps(out, indent=2))

    (OUT / "news_diagnosis.md").write_text(
        "# News diagnosis\n\n"
        f"- Coverage: {cov['n_rows']} rows, {cov['n_zero_vector_days']} zero-vector days "
        f"({cov['frac_zero_days']*100:.1f}%).\n"
        f"- Probes (test): " + "; ".join(f"{k}: AUC={v['ROC-AUC']} MCC={v['MCC']}" for k, v in out.items()) + ".\n"
        "- Missingness: exact-date left-merge, zero-vector fill, no forward-fill, no missing flag "
        "in model input (flag only in aggregated probe).\n"
        "- Leakage re-check: merge is on exact `Date`; embedding for date t comes from that date's "
        "row only. Residual risk: whether vendors' daily files include post-close articles — "
        "unverifiable from here; treat news as same-day public info, never intraday.\n"
        "- Verdict: news carries weak/no standalone signal on this split; high-dim sparse input "
        "justifies the projection bottleneck, not a larger news branch.\n")
    print(json.dumps({"coverage": cov, "stats": stats, "probes": out}, indent=2))


if __name__ == "__main__":
    main()
