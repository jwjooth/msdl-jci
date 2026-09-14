"""Robust multi-fold expanding-window walk-forward validation (audit Phase 4)."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from msdl_jci.evaluation.metrics import compute_classification_metrics
from msdl_jci.evaluation.trading_simulation import simulate_trading_strategy
from msdl_jci.evaluation.walk_forward import (
    EvaluationResults,
    MultiSourceTorchDataset,
    set_all_seeds,
    train_single_split,
)
from msdl_jci.utils.dataset_builder import MultiSourceTensors
from msdl_jci.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_range: tuple
    val_range: tuple
    test_range: tuple
    result: EvaluationResults


def expanding_window_splits(
    n: int, n_folds: int = 3, val_ratio: float = 0.10, embargo: int = 5
) -> list[tuple]:
    """Generate expanding-window (train grows, test slides) fold boundaries.

    Each fold reserves the last ``1/n_folds``-style test chunk; earlier folds use
    proportionally smaller histories. Returns (train_end, val_start, val_end, test_start, test_end).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    # Test chunks of equal size covering the final portion; train expands.
    test_size = max(n // (n_folds + 2), embargo + 10)
    splits = []
    for k in range(n_folds):
        test_end = n - (n_folds - 1 - k) * test_size
        test_start_raw = test_end - test_size
        val_size = max(int(n * val_ratio), embargo + 10)
        val_end_raw = test_start_raw - embargo
        val_start_raw = val_end_raw - val_size
        train_end_raw = val_start_raw - embargo
        if train_end_raw < 50 or val_start_raw < 0:
            continue
        splits.append((train_end_raw, val_start_raw, val_end_raw, test_start_raw, test_end))
    if not splits:
        # Fallback: single split mirroring evaluate_model_walk_forward defaults.
        train_end = int(n * 0.70)
        val_end = int(n * 0.80)
        splits.append((train_end, train_end + embargo, val_end, val_end + embargo, n))
    return splits


def run_robust_walk_forward(
    model_fn: Callable,
    model_name: str,
    tensors: MultiSourceTensors,
    n_folds: int = 3,
    embargo: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
    use_class_weight: bool = True,
) -> dict:
    """Run multi-fold expanding-window evaluation with aggregate stats.

    Returns dict with per-fold results + aggregate mean/std/CIs for AUC,
    balanced accuracy, MCC, and Sharpe.
    """
    set_all_seeds(seed)
    n = len(tensors.y)
    splits = expanding_window_splits(n, n_folds=n_folds, embargo=embargo)
    logger.info("Robust WF: %d folds for %s (n=%d, embargo=%d)", len(splits), model_name, n, embargo)

    fold_results: list[FoldResult] = []
    for i, (tr_end, v_start, v_end, te_start, te_end) in enumerate(splits):
        logger.info(
            "Fold %d: train[:%d] val[%d:%d] test[%d:%d]",
            i + 1, tr_end, v_start, v_end, te_start, te_end,
        )
        train_ds = MultiSourceTorchDataset(
            tensors.X_tech[:tr_end], tensors.X_macro[:tr_end], tensors.X_news[:tr_end],
            tensors.y[:tr_end], tensors.returns_5d[:tr_end],
        )
        val_ds = MultiSourceTorchDataset(
            tensors.X_tech[v_start:v_end], tensors.X_macro[v_start:v_end],
            tensors.X_news[v_start:v_end], tensors.y[v_start:v_end],
            tensors.returns_5d[v_start:v_end],
        )
        test_ds = MultiSourceTorchDataset(
            tensors.X_tech[te_start:te_end], tensors.X_macro[te_start:te_end],
            tensors.X_news[te_start:te_end], tensors.y[te_start:te_end],
            tensors.returns_5d[te_start:te_end],
        )
        model = model_fn()
        _, probs, weights, best_thr, history = train_single_split(
            model, train_ds, val_ds, test_ds,
            epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
            seed=seed + i, use_class_weight=use_class_weight, verbose=False,
        )
        clf = compute_classification_metrics(tensors.y[te_start:te_end], probs, threshold=best_thr)
        fin = simulate_trading_strategy(
            probs, tensors.returns_5d[te_start:te_end],
            dates=tensors.dates[te_start:te_end], threshold=best_thr,
        )
        res = EvaluationResults(
            model_name=f"{model_name} [fold {i+1}]",
            classification_metrics=clf, financial_metrics=fin,
            predictions=probs, actuals=tensors.y[te_start:te_end],
            test_dates=tensors.dates[te_start:te_end],
            gating_weights=weights, best_threshold=best_thr, history=history,
        )
        fold_results.append(FoldResult(i + 1, (0, tr_end), (v_start, v_end), (te_start, te_end), res))

    def _collect(attr_fn: Callable[[EvaluationResults], float]) -> np.ndarray:
        return np.array([attr_fn(f.result) for f in fold_results], dtype=float)

    aucs = _collect(lambda r: r.classification_metrics.roc_auc)
    balacc = _collect(lambda r: r.classification_metrics.balanced_accuracy)
    mccs = _collect(lambda r: r.classification_metrics.mcc)
    sharpes = _collect(lambda r: r.financial_metrics.sharpe_ratio)

    def _stats(x: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(x)) if len(x) else float("nan"),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "min": float(np.min(x)) if len(x) else float("nan"),
            "max": float(np.max(x)) if len(x) else float("nan"),
        }

    roc_stat, bal_stat, mcc_stat, shr_stat = (
        _stats(aucs), _stats(balacc), _stats(mccs), _stats(sharpes),
    )
    summary: dict[str, object] = {
        "model": model_name,
        "n_folds": len(fold_results),
        "roc_auc": roc_stat,
        "balanced_accuracy": bal_stat,
        "mcc": mcc_stat,
        "sharpe": shr_stat,
        "folds": fold_results,
    }
    logger.info(
        "Robust WF %s: AUC=%.4f±%.4f balAcc=%.4f±%.4f MCC=%.4f±%.4f Sharpe=%.3f±%.3f",
        model_name, roc_stat["mean"], roc_stat["std"],
        bal_stat["mean"], bal_stat["std"],
        mcc_stat["mean"], mcc_stat["std"],
        shr_stat["mean"], shr_stat["std"],
    )
    return summary
