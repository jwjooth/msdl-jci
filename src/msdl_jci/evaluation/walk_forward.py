"""Walk-Forward Validation and Training Engine for MSDL-JCI.

Implements the expanding-window chronological validation scheme (Thesis Section 2.5 & 2.6):
- Strictly respects time-series ordering to prevent look-ahead bias.
- Early stopping based on validation split to prevent overfitting.
- Evaluates both classification metrics and financial backtest metrics.
- Collects dynamic gating weights (alpha, beta, gamma) across regimes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from msdl_jci.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from msdl_jci.evaluation.trading_simulation import FinancialMetrics, simulate_trading_strategy
from msdl_jci.utils.dataset_builder import MultiSourceTensors
from msdl_jci.utils.logging_config import get_logger

logger = get_logger(__name__)


class MultiSourceTorchDataset(Dataset):
    """PyTorch Dataset wrapping multi-source modalities."""

    def __init__(
        self,
        X_tech: np.ndarray,
        X_macro: np.ndarray,
        X_news: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        self.X_tech = torch.tensor(X_tech, dtype=torch.float32)
        self.X_macro = torch.tensor(X_macro, dtype=torch.float32)
        self.X_news = torch.tensor(X_news, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X_tech[idx],
            self.X_macro[idx],
            self.X_news[idx],
            self.y[idx],
            self.returns[idx],
        )


@dataclass
class EvaluationResults:
    """Consolidated experimental results for a model run."""

    model_name: str
    classification_metrics: ClassificationMetrics
    financial_metrics: FinancialMetrics
    predictions: np.ndarray          # Predicted probabilities [N_test]
    actuals: np.ndarray              # Ground truth labels [N_test]
    test_dates: np.ndarray           # Dates of test period [N_test]
    gating_weights: Optional[np.ndarray] = None  # [N_test, 3] if model produces weights


def train_single_split(
    model: nn.Module,
    train_dataset: MultiSourceTorchDataset,
    val_dataset: MultiSourceTorchDataset,
    test_dataset: MultiSourceTorchDataset,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, np.ndarray, Optional[np.ndarray]]:
    """Train a model on train_dataset with early stopping on val_dataset, and predict on test_dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val_loss = float("inf")
    best_weights = None
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        # 1. Train
        model.train()
        train_loss = 0.0
        for b_tech, b_macro, b_news, b_y, _ in train_loader:
            b_tech = b_tech.to(device)
            b_macro = b_macro.to(device)
            b_news = b_news.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            logits, _ = model(b_tech, b_macro, b_news)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(b_y)

        train_loss /= len(train_dataset)

        # 2. Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_tech, b_macro, b_news, b_y, _ in val_loader:
                b_tech = b_tech.to(device)
                b_macro = b_macro.to(device)
                b_news = b_news.to(device)
                b_y = b_y.to(device)

                logits, _ = model(b_tech, b_macro, b_news)
                loss = criterion(logits, b_y)
                val_loss += loss.item() * len(b_y)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.debug("Early stopping triggered at epoch %d", epoch)
                break

    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 3. Test Inference
    model.eval()
    all_probs = []
    all_weights = []

    with torch.no_grad():
        for b_tech, b_macro, b_news, _, _ in test_loader:
            b_tech = b_tech.to(device)
            b_macro = b_macro.to(device)
            b_news = b_news.to(device)

            logits, weights = model(b_tech, b_macro, b_news)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

            if weights is not None:
                all_weights.append(weights.cpu().numpy())

    test_probs = np.array(all_probs)
    test_weights = np.vstack(all_weights) if all_weights else None
    return model, test_probs, test_weights


def evaluate_model_walk_forward(
    model_fn: callable,
    model_name: str,
    tensors: MultiSourceTensors,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
) -> EvaluationResults:
    """Execute expanding window validation and generate thesis evaluation metrics.

    Partitions:
    - Train set: First 70% of chronological data
    - Validation set: Next 10% (for early stopping & tuning)
    - Test set: Final 20% of chronological data (unseen future)
    """
    n_samples = len(tensors.y)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    logger.info(
        "Split sizes for %s: Train=%d, Val=%d, Test=%d (Total=%d)",
        model_name,
        train_end,
        val_end - train_end,
        n_samples - val_end,
        n_samples,
    )

    train_ds = MultiSourceTorchDataset(
        tensors.X_tech[:train_end],
        tensors.X_macro[:train_end],
        tensors.X_news[:train_end],
        tensors.y[:train_end],
        tensors.returns_5d[:train_end],
    )
    val_ds = MultiSourceTorchDataset(
        tensors.X_tech[train_end:val_end],
        tensors.X_macro[train_end:val_end],
        tensors.X_news[train_end:val_end],
        tensors.y[train_end:val_end],
        tensors.returns_5d[train_end:val_end],
    )
    test_ds = MultiSourceTorchDataset(
        tensors.X_tech[val_end:],
        tensors.X_macro[val_end:],
        tensors.X_news[val_end:],
        tensors.y[val_end:],
        tensors.returns_5d[val_end:],
    )

    model = model_fn()
    _, test_probs, test_weights = train_single_split(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )

    test_actuals = tensors.y[val_end:]
    test_returns = tensors.returns_5d[val_end:]
    test_dates = tensors.dates[val_end:]

    # Compute Classification Metrics (Thesis Section 2.7)
    clf_metrics = compute_classification_metrics(test_actuals, test_probs)

    # Compute Financial / Trading Simulation Metrics (Thesis Section 2.7)
    fin_metrics = simulate_trading_strategy(
        y_prob=test_probs,
        returns_5d=test_returns,
        dates=test_dates,
    )

    return EvaluationResults(
        model_name=model_name,
        classification_metrics=clf_metrics,
        financial_metrics=fin_metrics,
        predictions=test_probs,
        actuals=test_actuals,
        test_dates=test_dates,
        gating_weights=test_weights,
    )
