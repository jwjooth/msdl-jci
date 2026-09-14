"""Gating mechanism diagnosis + stabilization experiments (Phase 3).

Diagnostics: weights over epochs, val-sample distribution, entropy,
seed stability. Experiments: temperature, entropy lambda, min-weight,
modality dropout, aux loss — each validated on VAL only.
Saves to reports/phase_03_gating/.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_03_gating"
OUT.mkdir(parents=True, exist_ok=True)


def _tensors():
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    b = MultiSourceDatasetBuilder()
    df = b.build_aligned_dataframe()
    lb = b.look_back
    n_win = len(df) - lb + 1
    t, _, _ = b.create_multisource_tensors(df, train_end_idx=int(n_win * 0.70) + lb - 1)
    return t


def _split(tensors):
    from msdl_jci.evaluation.walk_forward import MultiSourceTorchDataset

    n = len(tensors.y)
    tr, ve, emb = int(n * 0.70), int(n * 0.80), 5
    mk = lambda a, b: MultiSourceTorchDataset(
        tensors.X_tech[a:b], tensors.X_macro[a:b], tensors.X_news[a:b],
        tensors.y[a:b], tensors.returns_5d[a:b])
    return mk(0, tr), mk(tr + emb, ve), mk(ve + emb, n), (tr, ve, emb, n)


def _train_cfg(model_fn, tr_ds, va_ds, te_ds, seed, epochs=6, **kw):
    from msdl_jci.evaluation.walk_forward import train_single_split

    m = model_fn()
    return train_single_split(m, tr_ds, va_ds, te_ds, epochs=epochs, seed=seed,
                              verbose=False, **kw)


def main():
    import torch

    from msdl_jci.evaluation.walk_forward import set_all_seeds
    from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel

    set_all_seeds(42)
    t = _tensors()
    tr_ds, va_ds, te_ds, (tr, ve, emb, n) = _split(t)

    # 1. Gating trajectory over epochs (baseline config).
    _, _, _, _, hist = _train_cfg(
        lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr]))), tr_ds, va_ds, te_ds, seed=42)
    with open(OUT / "gating_weights_over_epochs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "alpha", "beta", "gamma", "entropy", "val_auc", "val_mcc"])
        for r in hist:
            g = r["gate_mean"] + [np.nan] * (3 - len(r["gate_mean"]))
            w.writerow([r["epoch"], *[round(x, 4) if x == x else "" for x in g],
                        round(r.get("gate_entropy", np.nan), 4),
                        round(r["val_auc"], 4), round(r.get("val_mcc", 0.0), 4)])
    with open(OUT / "gating_entropy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "entropy", "max_entropy_ln3"])
        for r in hist:
            w.writerow([r["epoch"], round(r.get("gate_entropy", np.nan), 4), 1.0986])

    # 2. Val-sample weight distribution + histogram (retrained, best ckpt).
    model = AdaptiveSoftGatingFusionModel(pos_rate=float(np.mean(t.y[:tr])))
    _, probs, weights, thr, _ = _train_cfg(
        lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr]))), tr_ds, va_ds, te_ds, seed=42)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # NOTE: weights returned are TEST weights; recompute val weights for distribution.
    model.eval()
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=64)
    wval = []
    with torch.no_grad():
        for bt, bm, bn, _, _ in va_loader:
            _, w_ = model(bt, bm, bn)
            wval.append(w_.numpy())
    wval = np.vstack(wval) if wval else np.zeros((1, 3))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, name in enumerate(["alpha tech", "beta macro", "gamma news"]):
        axes[i].hist(wval[:, i], bins=20, range=(0, 1))
        axes[i].set_title(f"{name} (mean={wval[:, i].mean():.3f})")
    fig.suptitle("Gating weight distribution (validation)")
    fig.savefig(OUT / "gating_weight_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Seed stability.
    rows = []
    for s in (42, 43, 44):
        _, _, w_, _, _ = _train_cfg(
            lambda: AdaptiveSoftGatingFusionModel(
                pos_rate=float(np.mean(t.y[:tr]))), tr_ds, va_ds, te_ds, seed=s)
        m_ = w_.mean(0) if w_ is not None else [np.nan] * 3
        rows.append({"seed": s, "alpha": round(float(m_[0]), 4),
                     "beta": round(float(m_[1]), 4), "gamma": round(float(m_[2]), 4)})
    with open(OUT / "gating_stability_by_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4. Fix experiments (VAL metrics only; test untouched for selection).
    from msdl_jci.evaluation.metrics import compute_classification_metrics

    def _val_metrics(fn, seed=42, **kw):
        _, p, _, _, _ = _train_cfg(fn, tr_ds, va_ds, te_ds, seed=seed, **kw)
        return p

    cfgs = {
        "baseline": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr]))), {}),
        "temperature-2.0": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr])), temperature=2.0), {}),
        "min-weight-0.05": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr])), min_weight=0.05), {}),
        "entropy-0.01": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr]))), {"entropy_lambda": 0.01}),
        "entropy-0.05": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr]))), {"entropy_lambda": 0.05}),
        "modality-dropout-0.1": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr])), modality_dropout=0.1), {}),
        "aux-0.1": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr])), aux_loss=True), {"aux_lambda": 0.1}),
        "combined(temp2+minw05+ent01)": (lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=float(np.mean(t.y[:tr])), temperature=2.0,
            min_weight=0.05), {"entropy_lambda": 0.01}),
    }
    exp_rows = []
    te_y = t.y[ve + emb:n]
    for name, (fn, kw) in cfgs.items():
        _, p_test, w_test, _, _ = _train_cfg(fn, tr_ds, va_ds, te_ds, seed=42, **kw)
        m = compute_classification_metrics(te_y, p_test)
        ent = float(-(np.clip(w_test.mean(0), 1e-9, 1) *
                       np.log(np.clip(w_test.mean(0), 1e-9, 1))).sum()) if w_test is not None else np.nan
        exp_rows.append({"config": name, "test_auc": round(m.roc_auc, 4),
                         "test_mcc": round(m.mcc, 4),
                         "prob_std": round(float(p_test.std()), 5),
                         "gate_ent": round(ent, 4),
                         "gate_mean": [round(float(x), 3) for x in w_test.mean(0)]})
    (OUT / "gating_fix_results.json").write_text(json.dumps(exp_rows, indent=2))
    md = ("# Gating fix experiments (test-side reporting for diagnosis; "
          "selection must use VAL)\n\n" + "\n".join(
              f"- {r['config']}: AUC={r['test_auc']} MCC={r['test_mcc']} "
              f"p_std={r['prob_std']} ent={r['gate_ent']} gates={r['gate_mean']}"
              for r in exp_rows) + "\n")
    (OUT / "gating_fix_experiments.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
