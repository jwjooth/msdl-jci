"""Robust walk-forward + seed validation (Phase 9).

Models x seeds x expanding folds (embargo=5). Saves fold_seed_results.csv,
aggregate_results.csv, statistical_tests.md, stability_plots.png, predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_09_robust_validation"
OUT.mkdir(parents=True, exist_ok=True)
EPOCHS = 5
SEEDS = (42, 43, 44)


def main():
    from msdl_jci.evaluation.metrics import compute_classification_metrics
    from msdl_jci.evaluation.robust_walk_forward import expanding_window_splits
    from msdl_jci.evaluation.trading_simulation import simulate_trading_strategy
    from msdl_jci.evaluation.walk_forward import (
        MultiSourceTorchDataset,
        set_all_seeds,
        train_single_split,
    )
    from msdl_jci.models.fusion import (
        AdaptiveSoftGatingFusionModel,
        LSTMMacroModel,
        LSTMNewsModel,
        PureLSTMModel,
    )
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    b = MultiSourceDatasetBuilder()
    df = b.build_aligned_dataframe()
    lb = b.look_back
    n_win = len(df) - lb + 1
    t, _, _ = b.create_multisource_tensors(df, train_end_idx=int(n_win * 0.70) + lb - 1)
    n = len(t.y)
    pos_rate = float(np.mean(t.y[: int(n * 0.70)]))

    models = {
        "PureLSTM": lambda: PureLSTMModel(),
        "LSTM+Macro": lambda: LSTMMacroModel(),
        "LSTM+News": lambda: LSTMNewsModel(),
        "Proposed-stabilized": lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=pos_rate, temperature=2.0, min_weight=0.05,
            modality_dropout=0.05),
    }
    splits = expanding_window_splits(n, n_folds=2, embargo=5)
    rows = []
    curves = {}
    for mname, mfn in models.items():
        for seed in SEEDS:
            for fi, (tr_end, v_st, v_en, te_st, te_en) in enumerate(splits):
                set_all_seeds(seed)
                mk = lambda a, c: MultiSourceTorchDataset(
                    t.X_tech[a:c], t.X_macro[a:c], t.X_news[a:c],
                    t.y[a:c], t.returns_5d[a:c])
                m = mfn()
                _, probs, _, thr, _ = train_single_split(
                    m, mk(0, tr_end), mk(v_st, v_en), mk(te_st, te_en),
                    epochs=EPOCHS, seed=seed, verbose=False, entropy_lambda=0.01)
                yte = t.y[te_st:te_en]
                clf = compute_classification_metrics(yte, probs, threshold=thr)
                fin = simulate_trading_strategy(probs, t.returns_5d[te_st:te_en],
                                                dates=t.dates[te_st:te_en], threshold=thr)
                rows.append({"model": mname, "seed": seed, "fold": fi + 1,
                             "train_n": tr_end, "val_n": v_en - v_st,
                             "test_n": te_en - te_st, "threshold": round(thr, 2),
                             "auc": round(clf.roc_auc, 4),
                             "bal_acc": round(clf.balanced_accuracy, 4),
                             "mcc": round(clf.mcc, 4),
                             "logloss": round(clf.logloss, 4),
                             "prob_std": round(float(probs.std()), 5),
                             "net": round(fin.total_return, 2),
                             "gross": round(fin.gross_return, 2),
                             "sharpe": round(fin.sharpe_ratio, 3),
                             "maxdd": round(fin.max_drawdown, 2),
                             "exposure": round(fin.exposure_pct, 2),
                             "costs": round(fin.total_costs, 2)})
                curves.setdefault(mname, []).append(
                    {"seed": seed, "fold": fi + 1, "probs": probs,
                     "y": yte, "ret": t.returns_5d[te_st:te_en], "thr": thr})
                print(f"{mname} seed={seed} fold={fi+1}: auc={clf.roc_auc:.4f} "
                      f"mcc={clf.mcc:.4f} pstd={probs.std():.5f} net={fin.total_return:.2f}",
                      flush=True)
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "fold_seed_results.csv", index=False)
    agg = df_rows.groupby("model").agg(
        {c: ["mean", "std", "median", "min", "max"]
         for c in ("auc", "bal_acc", "mcc", "prob_std", "net", "gross",
                   "sharpe", "maxdd", "exposure", "costs")}).round(4)
    agg.to_csv(OUT / "aggregate_results.csv")

    # Statistical tests: paired Wilcoxon + t-test on fold AUC vs Proposed.
    from scipy.stats import ttest_rel, wilcoxon

    lines = [f"# Statistical tests (fold-level AUC, n={len(SEEDS) * len(splits)} per model)\n"]
    base = df_rows[df_rows.model == "Proposed-stabilized"]["auc"].values
    for m in df_rows.model.unique():
        if m == "Proposed-stabilized":
            continue
        x = df_rows[df_rows.model == m]["auc"].values
        try:
            _, pw = wilcoxon(x, base)
        except Exception as e:
            pw = f"n/a ({e})"
        try:
            _, pt = ttest_rel(x, base)
        except Exception as e:
            pt = f"n/a ({e})"
        lines.append(f"- {m} vs Proposed-stabilized: Wilcoxon p={pw}, paired-t p={pt} "
                     f"(mean {x.mean():.4f} vs {base.mean():.4f})")
    # Bootstrap CI for net-return difference (LSTM+Macro - Proposed).
    rng = np.random.default_rng(0)
    a = df_rows[df_rows.model == "LSTM+Macro"]["net"].values
    c = df_rows[df_rows.model == "Proposed-stabilized"]["net"].values
    diffs = [rng.choice(a, len(a), replace=True).mean() -
             rng.choice(c, len(c), replace=True).mean() for _ in range(2000)]
    lines.append(f"- Bootstrap 95% CI for (LSTM+Macro − Proposed) net return: "
                 f"[{np.quantile(diffs, .025):.2f}, {np.quantile(diffs, .975):.2f}]")
    (OUT / "statistical_tests.md").write_text("\n".join(lines) + "\n")

    # Stability plot: mean±std AUC and net return per model.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df_rows.groupby("model")[["auc", "net"]].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models_sorted = list(df_rows.model.unique())
    axes[0].errorbar(models_sorted, g["auc"]["mean"], yerr=g["auc"]["std"],
                     fmt="o", capsize=4)
    axes[0].axhline(0.5, color="gray", ls="--")
    axes[0].set_title("AUC mean±std (folds×seeds)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].errorbar(models_sorted, g["net"]["mean"], yerr=g["net"]["std"],
                     fmt="o", capsize=4)
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set_title("Net return % mean±std")
    axes[1].tick_params(axis="x", rotation=20)
    fig.savefig(OUT / "stability_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(OUT / "fold_predictions.npz",
                        **{f"{m}_{i}": np.stack([c_["y"], c_["probs"], c_["ret"]])
                           for m, cl in curves.items() for i, c_ in enumerate(cl)})
    print(df_rows.to_string(index=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
