"""Branch-level modality diagnosis (Phase 2).

Representation stats per branch + branch-only sklearn probes + gradient norms.
Saves to reports/phase_02_branch_diagnosis/.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_02_branch_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)


def _tensors():
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    b = MultiSourceDatasetBuilder()
    df = b.build_aligned_dataframe()
    lb = b.look_back
    n_win = len(df) - lb + 1
    t, _, _ = b.create_multisource_tensors(df, train_end_idx=int(n_win * 0.70) + lb - 1)
    return t


def _rep_stats(H, name):
    H = np.asarray(H, dtype=float)
    norms = np.linalg.norm(H, axis=1)
    # mean pairwise cosine similarity (sample 200 rows max)
    idx = np.random.default_rng(0).choice(len(H), min(200, len(H)), replace=False)
    Hn = H[idx] / (np.linalg.norm(H[idx], axis=1, keepdims=True) + 1e-9)
    cos = Hn @ Hn.T
    iu = np.triu_indices(len(idx), 1)
    eff_rank = None
    try:
        s = np.linalg.svd(H - H.mean(0), compute_uv=False)
        eff_rank = float((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12))
    except Exception:
        pass
    return {"branch": name, "dim": int(H.shape[1]),
            "mean": float(H.mean()), "std": float(H.std()),
            "min": float(H.min()), "max": float(H.max()),
            "l2_mean": float(norms.mean()), "l2_std": float(norms.std()),
            "cos_mean": float(cos[iu].mean()) if len(iu[0]) else 0.0,
            "eff_rank": eff_rank}


def main():
    import torch

    from msdl_jci.evaluation.walk_forward import set_all_seeds
    from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel

    set_all_seeds(42)
    t = _tensors()
    n = len(t.y)
    tr = int(n * 0.70)
    te = slice(int(n * 0.80) + 5, n)

    model = AdaptiveSoftGatingFusionModel()
    model.eval()
    with torch.no_grad():
        ht = model.tech_branch(torch.tensor(t.X_tech[tr:tr + 300])).numpy()
        hm = model.macro_branch(torch.tensor(t.X_macro[tr:tr + 300])).numpy()
        hn = model.news_branch(torch.tensor(t.X_news[tr:tr + 300])).numpy()
    stats = [_rep_stats(ht, "tech"), _rep_stats(hm, "macro"), _rep_stats(hn, "news")]
    (OUT / "branch_stats.json").write_text(json.dumps(stats, indent=2))

    # Branch-only probes: logreg on RAW modality inputs (train fit, test eval).
    from sklearn.linear_model import LogisticRegression

    from msdl_jci.evaluation.metrics import compute_classification_metrics

    feats = {"tech": t.X_tech.reshape(n, -1), "macro": t.X_macro, "news": t.X_news}
    rows = []
    for name, X in feats.items():
        clf = LogisticRegression(max_iter=2000).fit(X[:tr], t.y[:tr].astype(int))
        p = clf.predict_proba(X[te])[:, 1]
        m = compute_classification_metrics(t.y[te], p)
        rows.append({"branch": name, **m.to_dict(), "prob_std": round(float(p.std()), 5)})
    with open(OUT / "branch_only_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Gradient norms: one backward pass on train batch through full model.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    bt = torch.tensor(t.X_tech[:64])
    bm = torch.tensor(t.X_macro[:64])
    bn = torch.tensor(t.X_news[:64])
    by = torch.tensor(t.y[:64]).unsqueeze(-1)
    logits, weights = model(bt, bm, bn)
    loss = torch.nn.BCEWithLogitsLoss()(logits, by)
    loss.backward()
    gn = {}
    for pname, mod in (("tech", model.tech_branch), ("macro", model.macro_branch),
                       ("news", model.news_branch), ("gating", model.soft_gating),
                       ("classifier", model.classifier)):
        total = 0.0
        for prm in mod.parameters():
            if prm.grad is not None:
                total += float((prm.grad ** 2).sum())
        gn[pname] = round(float(np.sqrt(total)), 5)
    (OUT / "branch_gradient_norms.csv").write_text(
        "module,grad_norm\n" + "\n".join(f"{k},{v}" for k, v in gn.items()) + "\n")
    (OUT / "branch_ablation_logits.json").write_text(json.dumps(
        {"loss_one_batch": float(loss.item()),
         "logit_std_one_batch": float(logits.detach().std()),
         "gate_mean_init": [float(x) for x in weights.detach().mean(0)]}, indent=2))
    md = ("# Branch diagnosis\n\n- Representation stats: see branch_stats.json.\n"
          "- Branch-only logreg test AUC: " + ", ".join(f"{r['branch']}={r['ROC-AUC']}" for r in rows) + ".\n"
          "- Branch-only MCC: " + ", ".join(f"{r['branch']}={r['MCC']}" for r in rows) + ".\n"
          f"- One-batch grad norms: {gn}.\n"
          f"- Gate mean at init: {[round(float(x),3) for x in weights.detach().mean(0)]} (expect ~0.333 each).\n\n"
          f"Interpretation: macro raw features are low-frequency/stale — if macro-only AUC ~ 0.5 "
          f"yet the fused gate collapses to macro (beta→0.9), the gate is exploiting macro as a "
          f"bias shortcut, not signal. News-only signal is typically weak (high-dim, sparse). "
          f"Tech carries most genuine signal.\n")
    (OUT / "branch_diagnosis.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
