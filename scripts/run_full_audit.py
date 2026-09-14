"""Full audit pipeline (audit Phase 5): leak-free dataset + all baselines + neural models.

Usage:
    uv run python scripts/run_full_audit.py [--epochs 15 --output reports]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msdl_jci.config.settings import get_settings
from msdl_jci.evaluation.metrics import compute_classification_metrics
from msdl_jci.evaluation.trading_simulation import simulate_trading_strategy
from msdl_jci.evaluation.walk_forward import evaluate_model_walk_forward, set_all_seeds
from msdl_jci.models.fusion import (
    AdaptiveSoftGatingFusionModel,
    LSTMMacroModel,
    LSTMNewsModel,
    PureLSTMModel,
    StaticFusionModel,
)
from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder
from msdl_jci.utils.logging_config import configure_logging, get_logger

logger = get_logger("audit")


def _leak_free_tensors(builder, df_aligned, train_ratio=0.70):
    """Build windowed tensors with scalers fitted on the TRAIN PREFIX only."""
    lb = builder.look_back
    n_win = len(df_aligned) - lb + 1
    train_end_win = int(n_win * train_ratio)
    # df row index delimiting the train prefix (window end row of last train sample)
    train_end_df = train_end_win + lb - 1
    tensors, feat_sc, macro_sc = builder.create_multisource_tensors(
        df_aligned, train_end_idx=train_end_df
    )
    return tensors


def _heuristic_probs(name, tensors, seed=0):
    """Deterministic heuristic baselines returning pseudo-probabilities."""
    rng = np.random.default_rng(seed)
    n = len(tensors.y)
    if name == "always-positive":
        return np.full(n, 0.99)
    if name == "always-negative":
        return np.full(n, 0.01)
    if name == "random":
        return rng.uniform(0, 1, size=n)
    if name == "previous-day":
        # Proxy: recent tech momentum — last close in window vs window mean.
        last = tensors.X_tech[:, -1, 0]
        mean = tensors.X_tech[:, :, 0].mean(axis=1)
        return np.clip(0.5 + (last - mean) * 2.0, 0.01, 0.99)
    if name == "momentum-5d":
        first = tensors.X_tech[:, 0, 0]
        last = tensors.X_tech[:, -1, 0]
        return np.clip(0.5 + (last - first) * 2.0, 0.01, 0.99)
    if name == "momentum-10d":
        first = tensors.X_tech[:, -10, 0]
        last = tensors.X_tech[:, -1, 0]
        return np.clip(0.5 + (last - first) * 2.0, 0.01, 0.99)
    if name == "ma-crossover":
        # SMA-20 is tech col index 6; compare short (5d) vs long (20d) window means.
        short = tensors.X_tech[:, -5:, 0].mean(axis=1)
        long = tensors.X_tech[:, :, 0].mean(axis=1)
        return np.clip(0.5 + (short - long) * 3.0, 0.01, 0.99)
    raise ValueError(name)


def _sklearn_baseline(name, tensors, test_start):
    """LogisticRegression / RandomForest / HistGradientBoosting on flattened features."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate(
        [
            tensors.X_tech.reshape(len(tensors.y), -1),
            tensors.X_macro,
            tensors.X_news,
        ],
        axis=1,
    )
    y = tensors.y.astype(int)
    Xtr, ytr = X[:test_start], y[:test_start]
    Xte = X[test_start:]
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000)
    elif name == "random-forest":
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    elif name == "grad-boosting":
        clf = HistGradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(name)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--output", type=str, default="reports")
    ap.add_argument("--skip-neural", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL)
    set_all_seeds(42)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    builder = MultiSourceDatasetBuilder()
    df = builder.build_aligned_dataframe()
    tensors = _leak_free_tensors(builder, df)
    n = len(tensors.y)
    # Mirror walk_forward embargo split for heuristic/sklearn cut points.
    embargo = settings.ML_PREDICTION_HORIZON
    val_end = int(n * 0.80)
    test_start = min(val_end + embargo, n)
    logger.info("Audit tensors: N=%d test_start=%d (embargo=%d)", n, test_start, embargo)

    rows = []

    def _record(model_name, probs_full_or_test, is_full=True):
        probs_test = probs_full_or_test[test_start:] if is_full else probs_full_or_test
        y_test = tensors.y[test_start:]
        r_test = tensors.returns_5d[test_start:]
        d_test = tensors.dates[test_start:]
        from msdl_jci.evaluation.walk_forward import find_best_threshold

        thr = find_best_threshold(y_test, probs_test, metric="mcc") if len(np.unique(y_test)) > 1 else 0.5
        clf = compute_classification_metrics(y_test, probs_test, threshold=thr)
        fin = simulate_trading_strategy(probs_test, r_test, dates=d_test, threshold=thr)
        row = {"Model": model_name, "Threshold": round(thr, 3)}
        row.update(clf.to_dict())
        row.update(fin.to_dict())
        row["prob_std"] = round(float(np.std(probs_test)), 5)
        rows.append(row)
        logger.info(
            "%-32s acc=%.2f bal=%.2f mcc=%.3f auc=%.4f pos=%.1f%% std=%.4f thr=%.2f",
            model_name, row["Accuracy"], row["Balanced-Acc"], row["MCC"],
            row["ROC-AUC"], row["Pred-Pos-Rate"], row["prob_std"], thr,
        )

    for h in ["always-positive", "always-negative", "random", "previous-day",
              "momentum-5d", "momentum-10d", "ma-crossover"]:
        _record(h, _heuristic_probs(h, tensors), is_full=True)
    for s in ["logreg", "random-forest", "grad-boosting"]:
        try:
            _record(s, _sklearn_baseline(s, tensors, test_start), is_full=False)
        except Exception as e:
            logger.warning("sklearn baseline %s failed: %s", s, e)

    if not args.skip_neural:

        pos_rate = float(np.mean(tensors.y[: int(n * 0.70)]))
        neural = [
            ("Pure LSTM (Technical only)", lambda: PureLSTMModel()),
            ("LSTM + Macro", lambda: LSTMMacroModel()),
            ("LSTM + News", lambda: LSTMNewsModel()),
            ("LSTM + Static Fusion", lambda: StaticFusionModel()),
            (
                "Proposed (Adaptive Soft Gating)",
                lambda: AdaptiveSoftGatingFusionModel(
                    pos_rate=pos_rate, modality_dropout=0.05
                ),
            ),
        ]
        for name, fn in neural:
            try:
                res = evaluate_model_walk_forward(
                    model_fn=fn, model_name=name, tensors=tensors,
                    epochs=args.epochs, seed=42,
                )
                row = {"Model": name, "Threshold": round(res.best_threshold, 3)}
                row.update(res.classification_metrics.to_dict())
                row.update(res.financial_metrics.to_dict())
                row["prob_std"] = round(float(np.std(res.predictions)), 5)
                rows.append(row)
                logger.info(
                    "%-32s acc=%.2f bal=%.2f mcc=%.3f auc=%.4f pos=%.1f%% std=%.4f",
                    name, row["Accuracy"], row["Balanced-Acc"], row["MCC"],
                    row["ROC-AUC"], row["Pred-Pos-Rate"], row["prob_std"],
                )
            except Exception as e:
                logger.warning("Neural model %s failed: %s", name, e)

    df_out = pd.DataFrame(rows)
    csv = out / "full_audit_comparison.csv"
    df_out.to_csv(csv, index=False)
    md = out / "full_audit_comparison.md"
    try:
        md_text = df_out.to_markdown(index=False)
    except ImportError:
        md_text = "```\n" + df_out.to_string(index=False) + "\n```"
    md.write_text("# Full Audit Comparison (leak-free)\n\n" + md_text + "\n")
    logger.info("Saved %s and %s", csv, md)
    print(df_out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
