"""Evaluation diagnostics (audit Phase 6): histograms, ROC, calibration, confusion.

Usage:
    uv run python scripts/diagnose_evaluation.py --probs <npy> --labels <npy> [--out reports]
    # Or run without args for a quick synthetic smoke test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def diagnose(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path, tag: str = "eval") -> dict:
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    stats = {
        "n": len(y_true),
        "pos_rate": float(np.mean(y_true)),
        "prob_mean": float(np.mean(y_prob)),
        "prob_std": float(np.std(y_prob)),
        "prob_min": float(np.min(y_prob)),
        "prob_max": float(np.max(y_prob)),
        "pred_pos_rate@0.5": float(np.mean(y_prob >= 0.5)),
    }
    try:
        stats["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        stats["auc"] = 0.5
    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int), labels=[0, 1])
    stats["TN"], stats["FP"], stats["FN"], stats["TP"] = [int(x) for x in cm.ravel()]
    if stats["prob_std"] < 0.01:
        stats["warning"] = "COLLAPSE: near-constant predictions"
    elif stats["auc"] < 0.55:
        stats["warning"] = "Near-random discrimination (AUC<0.55)"
    else:
        stats["warning"] = "OK"

    # Probability histogram data
    hist, edges = np.histogram(y_prob, bins=20, range=(0, 1))
    lines = [f"# Diagnostic: {tag}", ""]
    lines.append(f"n={stats['n']} pos_rate={stats['pos_rate']:.3f} "
                 f"p_mean={stats['prob_mean']:.4f} p_std={stats['prob_std']:.5f} "
                 f"range=[{stats['prob_min']:.4f},{stats['prob_max']:.4f}]")
    lines.append(f"AUC={stats['auc']:.4f} CM=[TN={stats['TN']} FP={stats['FP']} FN={stats['FN']} TP={stats['TP']}]")
    lines.append(f"Verdict: {stats['warning']}")
    lines.append("")
    lines.append("Prob histogram (20 bins over [0,1]):")
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        lines.append(f"  [{e0:.2f},{e1:.2f}): {'#' * min(int(h), 80)} ({h})")
    # Calibration by decile
    lines.append("")
    lines.append("Calibration (decile: n, mean_p, emp_rate):")
    order = np.argsort(y_prob)
    for d in range(10):
        sl = order[d * len(order) // 10:(d + 1) * len(order) // 10]
        if len(sl):
            lines.append(f"  D{d}: n={len(sl)} mean_p={np.mean(y_prob[sl]):.3f} emp={np.mean(y_true[sl]):.3f}")
    # ROC points
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        lines.append("")
        lines.append("ROC points (FPR,TPR):")
        for f, t in zip(fpr[:: max(len(fpr) // 10, 1)], tpr[:: max(len(tpr) // 10, 1)]):
            lines.append(f"  ({f:.3f},{t:.3f})")
    except ValueError:
        pass
    (out_dir / f"diagnose_{tag}.txt").write_text("\n".join(lines) + "\n")
    # Save decile table CSV
    print("\n".join(lines))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=str, default=None)
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--out", type=str, default="reports")
    ap.add_argument("--tag", type=str, default="eval")
    args = ap.parse_args()

    out = Path(args.out)
    if args.probs and args.labels:
        probs = np.load(args.probs)
        labels = np.load(args.labels)
    else:
        rng = np.random.default_rng(0)
        labels = (rng.uniform(0, 1, 300) > 0.45).astype(int)
        probs = np.clip(0.5 + (labels - 0.5) * 0.1 + rng.normal(0, 0.02, 300), 0, 1)
        print("No --probs/--labels given: running synthetic smoke test.")
    diagnose(labels, probs, out, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
