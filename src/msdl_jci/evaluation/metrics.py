"""Classification metrics evaluation module for MSDL-JCI."""

from dataclasses import dataclass
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
    )
