"""Classification metrics evaluation module for MSDL-JCI."""

from dataclasses import dataclass
<<<<<<< HEAD

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Quantitative classification metrics matching Thesis Section 2.7.

    Extended (audit Phase 2) with collapse-sensitive metrics: balanced
    accuracy, MCC, PR-AUC, log-loss, Brier score.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]
    balanced_accuracy: float = 0.5
    mcc: float = 0.0
    pr_auc: float = 0.5
    logloss: float = float("nan")
    brier: float = float("nan")
    threshold: float = 0.5
    predicted_positive_rate: float = 0.0

    def to_dict(self) -> dict[str, float | list]:
        return {
            "Accuracy": round(self.accuracy * 100, 2),
            "Precision": round(self.precision * 100, 2),
            "Recall": round(self.recall * 100, 2),
            "F1-Score": round(self.f1_score * 100, 2),
            "ROC-AUC": round(self.roc_auc, 4),
            "Balanced-Acc": round(self.balanced_accuracy * 100, 2),
            "MCC": round(float(self.mcc), 4),
            "PR-AUC": round(float(self.pr_auc), 4),
            "LogLoss": round(float(self.logloss), 4) if np.isfinite(self.logloss) else float("nan"),
            "Brier": round(float(self.brier), 4) if np.isfinite(self.brier) else float("nan"),
            "Threshold": round(float(self.threshold), 3),
            "Pred-Pos-Rate": round(float(self.predicted_positive_rate) * 100, 2),
            "TN": int(self.confusion_matrix[0, 0]),
            "FP": int(self.confusion_matrix[0, 1]),
            "FN": int(self.confusion_matrix[1, 0]),
            "TP": int(self.confusion_matrix[1, 1]),
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute standard + collapse-sensitive classification evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_prob: Predicted probabilities for class 1 (UP)
        threshold: Decision threshold for classification (default 0.5)

    Returns:
        ClassificationMetrics instance
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    try:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        bal_acc = 0.5

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
        if np.isnan(mcc):
            mcc = 0.0
    except ValueError:
        mcc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
        if np.isnan(pr_auc):
            pr_auc = float(np.mean(y_true))
    except ValueError:
        pr_auc = 0.5

    try:
        ll = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)))
    except ValueError:
        ll = float("nan")

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        brier = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pos_rate = float(np.mean(y_pred)) if len(y_pred) else 0.0

    return ClassificationMetrics(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1_score=float(f1),
        roc_auc=float(auc),
        confusion_matrix=cm,
        balanced_accuracy=float(bal_acc),
        mcc=float(mcc),
        pr_auc=float(pr_auc),
        logloss=float(ll),
        brier=float(brier),
        threshold=float(threshold),
        predicted_positive_rate=pos_rate,
||||||| c9f8f7e
=======
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Quantitative classification metrics matching Thesis Section 2.7."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]

    def to_dict(self) -> Dict[str, Union[float, list]]:
        return {
            "Accuracy": round(self.accuracy * 100, 2),
            "Precision": round(self.precision * 100, 2),
            "Recall": round(self.recall * 100, 2),
            "F1-Score": round(self.f1_score * 100, 2),
            "ROC-AUC": round(self.roc_auc, 4),
            "TN": int(self.confusion_matrix[0, 0]),
            "FP": int(self.confusion_matrix[0, 1]),
            "FN": int(self.confusion_matrix[1, 0]),
            "TP": int(self.confusion_matrix[1, 1]),
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute standard classification evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_prob: Predicted probabilities for class 1 (UP)
        threshold: Decision threshold for classification (default 0.5)

    Returns:
        ClassificationMetrics instance
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return ClassificationMetrics(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1_score=float(f1),
        roc_auc=float(auc),
        confusion_matrix=cm,
>>>>>>> main
    )
