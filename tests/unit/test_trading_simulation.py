"""Unit tests for trading simulation and metrics."""

import numpy as np
import pytest
from msdl_jci.evaluation.metrics import compute_classification_metrics
from msdl_jci.evaluation.trading_simulation import simulate_trading_strategy


def test_compute_classification_metrics():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.4, 0.6, 0.3])
    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics.accuracy == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix[0, 0] == 4  # TN
    assert metrics.confusion_matrix[1, 1] == 4  # TP


def test_simulate_trading_strategy():
    n = 20
    # Alternating returns
    returns = np.array([0.02, -0.01, 0.03, -0.02] * 5)
    # Perfect model signals
    y_prob = np.array([0.9, 0.1, 0.8, 0.2] * 5)

    res = simulate_trading_strategy(y_prob, returns, stride=1)
    assert res.total_return > 0.0
    assert res.win_rate == 100.0  # avoided all negative returns
    assert res.max_drawdown >= 0.0
    assert res.sharpe_ratio > 0.0
