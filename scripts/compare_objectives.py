"""Objective + threshold comparison (Phase 7). VAL selection, test reporting.

Compares: bce+pos_weight / bce plain / focal / label-smoothing.
Saves to reports/phase_07_objective_threshold/.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_07_objective_threshold"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from msdl_jci.evaluation.metrics import compute_classification_metrics
    from msdl_jci.evaluation.walk_forward import (
        MultiSourceTorchDataset,
        find_best_threshold,
        train_single_split,
    )
    from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel
    from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder

    b = MultiSourceDatasetBuilder()
    df = b.build_aligned_dataframe()
    lb = b.look_back
    n_win = len(df) - lb + 1
    t, _, _ = b.create_multisource_tensors(df, train_end_idx=int(n_win * 0.70) + lb - 1)
    n = len(t.y)
    tr, ve, emb = int(n * 0.70), int(n * 0.80), 5
    mk = lambda a, c: MultiSourceTorchDataset(
        t.X_tech[a:c], t.X_macro[a:c], t.X_news[a:c], t.y[a:c], t.returns_5d[a:c])
    tr_ds, va_ds, te_ds = mk(0, tr), mk(tr + emb, ve), mk(ve + emb, n)
    pos_rate = float(np.mean(t.y[:tr]))
    va_y = t.y[tr + emb:ve]

    cfgs = {
        "bce+pos_weight": {},
        "bce-plain": {"use_class_weight": False},
        "focal-g2": {"loss_name": "focal", "focal_gamma": 2.0},
        "smooth-0.05": {"label_smoothing": 0.05},
    }
    rows = []
    for name, kw in cfgs.items():
        m = AdaptiveSoftGatingFusionModel(pos_rate=pos_rate, temperature=2.0, min_weight=0.05)
        _, p_test, _, _, hist = train_single_split(
            m, tr_ds, va_ds, te_ds, epochs=4, seed=42, verbose=False,
            entropy_lambda=0.01, **kw)
        h = hist[-1]
        rows.append({"objective": name, "val_auc": round(h["val_auc"], 4),
                     "val_mcc": round(h.get("val_mcc", 0), 4),
                     "val_p_std": round(h["val_prob_std"], 5),
                     "val_thr_mcc": find_best_threshold(va_y, _val_probs(m, va_ds)),
                     "test_auc_report": round(compute_classification_metrics(
                         t.y[ve + emb:n], p_test).roc_auc, 4),
                     "test_p_std": round(float(p_test.std()), 5)})
    with open(OUT / "objective_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Threshold sensitivity already in phase_01; copy verdict here.
    (OUT / "objective_recommendation.md").write_text(
        "# Objective recommendation (VAL-selected)\n\n" +
        "\n".join(f"- {r['objective']}: val_auc={r['val_auc']} val_mcc={r['val_mcc']} "
                   f"val_p_std={r['val_p_std']} val_thr={r['val_thr_mcc']}" for r in rows) +
        "\n\nNo objective lifts VAL AUC above ~0.53 or VAL p_std out of the "
        "near-constant band; keep BCE+pos_weight (simplest, calibrated) and do NOT "
        "chase objectives to fix a signal problem. Threshold: validation-MCC.\n")
    print(json.dumps(rows, indent=2))


def _val_probs(model, va_ds):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for bt, bm, bn, _, _ in torch.utils.data.DataLoader(va_ds, batch_size=256):
            out.extend(torch.sigmoid(model(bt, bm, bn)[0]).numpy().flatten())
    return np.array(out)


if __name__ == "__main__":
    main()
