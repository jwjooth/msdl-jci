"""Walk-Forward Validation and Training Engine for MSDL-JCI.

Implements the expanding-window chronological validation scheme (Thesis Section 2.5 & 2.6):
- Strictly respects time-series ordering to prevent look-ahead bias.
- Early stopping based on validation split to prevent overfitting.
- Evaluates both classification metrics and financial backtest metrics.
- Collects dynamic gating weights (alpha, beta, gamma) across regimes.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from msdl_jci.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from msdl_jci.evaluation.trading_simulation import FinancialMetrics, simulate_trading_strategy
from msdl_jci.utils.dataset_builder import MultiSourceTensors
from msdl_jci.utils.logging_config import get_logger

logger = get_logger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may cost speed, gains reproducibility)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Compute BCE pos_weight = neg/pos from the TRAINING labels only."""
    y = np.asarray(y_train).astype(float).flatten()
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos < 1:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Binary focal loss on logits (lin et al.). pos_weight applied like BCE."""

    smooth: float

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = pos_weight
        self.smooth = 0.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce.clamp_min(1e-9))
        return ((1.0 - pt) ** self.gamma * bce).mean()


def make_criterion(
    loss: str = "bce",
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build training criterion. Targets smoothed only if label_smoothing > 0."""
    if loss == "focal":
        mod = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        mod.smooth = float(label_smoothing)
        return mod
    base = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )
    if label_smoothing > 0.0:

        class _Smooth(nn.Module):
            def __init__(self, inner, eps):
                super().__init__()
                self.inner, self.eps = inner, float(eps)

            def forward(self, logits, targets):
                t = targets * (1 - 2 * self.eps) + self.eps
                return self.inner(logits, t)

        return _Smooth(base, label_smoothing)
    return base


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "mcc",
    grid: np.ndarray | None = None,
) -> float:
    """Search decision threshold on VALIDATION predictions.

    Maximizes MCC (default) or balanced accuracy. Grid-searches [0.2, 0.8].
    """
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

    if grid is None:
        grid = np.arange(0.20, 0.801, 0.01)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    best_thr, best_score = 0.5, -np.inf
    for thr in grid:
        y_pred = (y_prob >= thr).astype(int)
        try:
            if metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, y_pred)
            else:  # mcc
                score = matthews_corrcoef(y_true, y_pred)
                if np.isnan(score):
                    score = -np.inf
        except ValueError:
            score = -np.inf
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr


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

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    predictions: np.ndarray  # Predicted probabilities [N_test]
    actuals: np.ndarray  # Ground truth labels [N_test]
    test_dates: np.ndarray  # Dates of test period [N_test]
    gating_weights: np.ndarray | None = None  # [N_test, 3] if model produces weights
    best_threshold: float = 0.5  # Val-optimized decision threshold
    history: list[dict] | None = field(default=None)  # Per-epoch diagnostics


def _predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return (probs, gating_weights or None, true labels) for a loader."""
    model.eval()
    all_probs: list[float] = []
    all_weights: list[np.ndarray] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for b_tech, b_macro, b_news, b_y, _ in loader:
            logits, weights = model(b_tech.to(device), b_macro.to(device), b_news.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(b_y.cpu().numpy().flatten())
            if weights is not None:
                all_weights.append(weights.cpu().numpy())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    weights = np.vstack(all_weights) if all_weights else None
    return probs, weights, labels


def train_single_split(
    model: nn.Module,
    train_dataset: MultiSourceTorchDataset,
    val_dataset: MultiSourceTorchDataset,
    test_dataset: MultiSourceTorchDataset,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    device: torch.device | None = None,
    seed: int = 42,
    use_class_weight: bool = True,
    verbose: bool = True,
    entropy_lambda: float = 0.0,
    aux_lambda: float = 0.0,
    early_stop_metric: str = "loss",
    loss_name: str = "bce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> tuple[nn.Module, np.ndarray, np.ndarray | None, float, list[dict]]:
    """Train a model on train_dataset with early stopping on val_dataset, and predict on test_dataset.

    Fixes applied (audit Phase 1):
    - Deterministic seeding (Python/NumPy/Torch).
    - BCEWithLogitsLoss ``pos_weight`` from TRAIN labels only (anti-collapse).
    - Per-epoch diagnostics: train/val loss, val AUC, val balanced accuracy,
      predicted positive ratio, gradient norms, gating weight means.
    - Validation-threshold optimization (MCC) returned for unbiased test evaluation.

    Phase 3 extras:
    - ``entropy_lambda``: weight of gating-entropy bonus
      (total = main - lambda * entropy); validated on val, never test.
    - ``aux_lambda``: weight of per-branch auxiliary BCE (requires model with
      ``aux_logits()``); kept small so it cannot dominate.
    - ``early_stop_metric``: 'loss' (default), 'mcc', or 'auc'.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    set_all_seeds(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # Deterministic shuffling via generator seed.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Class weighting from TRAIN ONLY (prevents always-positive collapse) ---
    if use_class_weight:
        pos_weight = compute_pos_weight(
            train_dataset.y.cpu().numpy()
            if torch.is_tensor(train_dataset.y)
            else np.asarray(train_dataset.y)
        ).to(device)
        logger.info("BCE pos_weight (train neg/pos): %.4f", float(pos_weight.cpu()))
    else:
        pos_weight = None
    criterion: nn.Module = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )
    # Phase 7 objective overrides (validated on VAL, never test).
    if loss_name == "focal":
        criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    elif label_smoothing > 0.0:
        criterion = make_criterion("bce", pos_weight, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    has_aux = hasattr(model, "aux_logits") and aux_lambda > 0.0

    best_val_loss = float("inf")
    best_val_mcc = -np.inf
    best_val_auc = -np.inf
    best_weights = None
    early_stop_counter = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # 1. Train
        model.train()
        train_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        for b_tech, b_macro, b_news, b_y, _ in train_loader:
            b_tech = b_tech.to(device)
            b_macro = b_macro.to(device)
            b_news = b_news.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            logits, weights = model(b_tech, b_macro, b_news)
            loss = criterion(logits, b_y)
            if entropy_lambda > 0.0 and weights is not None:
                ent = -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean()
                loss = loss - entropy_lambda * ent
            if has_aux:
                aux_fn = getattr(model, "aux_logits", None)
                if callable(aux_fn):
                    try:
                        aux = aux_fn(b_tech, b_macro, b_news)  # [B,3]
                        aux_loss = criterion(aux, b_y.expand(-1, aux.shape[1]))
                        loss = loss + aux_lambda * aux_loss
                    except RuntimeError:
                        pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            try:
                grad_norm_f = float(grad_norm)
            except Exception:
                grad_norm_f = float("nan")
            total_grad_norm += grad_norm_f
            n_batches += 1
            optimizer.step()
            train_loss += loss.item() * len(b_y)

        train_loss /= max(len(train_dataset), 1)
        avg_grad_norm = total_grad_norm / max(n_batches, 1)

        # 2. Validate (with diagnostics)
        model.eval()
        val_loss = 0.0
        val_probs_list: list[float] = []
        val_labels_list: list[float] = []
        val_w_list: list[np.ndarray] = []
        with torch.no_grad():
            for b_tech, b_macro, b_news, b_y, _ in val_loader:
                b_tech = b_tech.to(device)
                b_macro = b_macro.to(device)
                b_news = b_news.to(device)
                b_y = b_y.to(device)

                logits, w = model(b_tech, b_macro, b_news)
                loss = criterion(logits, b_y)
                val_loss += loss.item() * len(b_y)
                val_probs_list.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels_list.extend(b_y.cpu().numpy().flatten())
                if w is not None:
                    val_w_list.append(w.cpu().numpy())

        val_loss /= max(len(val_dataset), 1)
        scheduler.step(val_loss)

        # Diagnostics
        val_probs = np.array(val_probs_list)
        val_labels = np.array(val_labels_list, dtype=int)
        try:
            val_auc = (
                float(roc_auc_score(val_labels, val_probs))
                if len(np.unique(val_labels)) > 1
                else 0.5
            )
        except ValueError:
            val_auc = 0.5
        try:
            val_bal_acc = float(balanced_accuracy_score(val_labels, (val_probs >= 0.5).astype(int)))
        except ValueError:
            val_bal_acc = 0.5
        try:
            from sklearn.metrics import matthews_corrcoef as _mcc

            _m = _mcc(val_labels, (val_probs >= 0.5).astype(int))
            val_mcc = float(_m) if not np.isnan(_m) else 0.0
        except ValueError:
            val_mcc = 0.0
        pos_ratio = float(np.mean(val_probs >= 0.5)) if len(val_probs) else 0.0
        prob_mean, prob_std = (
            float(np.mean(val_probs)) if len(val_probs) else 0.0,
            float(np.std(val_probs)) if len(val_probs) else 0.0,
        )
        if val_w_list:
            gw = np.vstack(val_w_list)
            gate_mean = [float(np.mean(gw[:, i])) for i in range(gw.shape[1])]
            _p = np.clip(gw.mean(0), 1e-9, 1.0)
            gate_entropy = float(-np.sum(_p * np.log(_p)))
        else:
            gate_mean = []
            gate_entropy = float("nan")

        rec = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": val_auc,
            "val_bal_acc": val_bal_acc,
            "val_mcc": val_mcc,
            "val_pos_ratio": pos_ratio,
            "val_prob_mean": prob_mean,
            "val_prob_std": prob_std,
            "grad_norm": avg_grad_norm,
            "gate_mean": gate_mean,
            "gate_entropy": gate_entropy,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(rec)
        if verbose:
            logger.info(
                "Epoch %02d/%d | train_loss=%.4f val_loss=%.4f | val_auc=%.4f "
                "val_balacc=%.4f val_mcc=%.4f pos_ratio=%.3f p_mean=%.4f p_std=%.4f | grad=%.4f ent=%.3f gates=%s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_auc,
                val_bal_acc,
                val_mcc,
                pos_ratio,
                prob_mean,
                prob_std,
                avg_grad_norm,
                gate_entropy,
                [round(x, 3) for x in gate_mean] if gate_mean else "n/a",
            )
        # Collapse warning: near-constant predictions are a training failure.
        if prob_std < 0.01 and epoch >= 5 and verbose:
            logger.warning(
                "Possible model collapse at epoch %d: val prob std=%.5f (near-constant output)",
                epoch,
                prob_std,
            )

        improved = False
        if early_stop_metric == "mcc":
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                improved = True
        elif early_stop_metric == "auc":
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                improved = True
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
        if improved:
            best_val_loss = min(best_val_loss, val_loss)
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

    # 3. Val-threshold optimization (MCC) on best checkpoint
    val_probs_best, _, val_labels_best = _predict_probs(model, val_loader, device)
    best_thr = find_best_threshold(val_labels_best, val_probs_best, metric="mcc")
    logger.info("Val-optimized threshold (MCC): %.2f", best_thr)

    # 4. Test Inference
    model.eval()
    all_probs2: list[float] = []
    all_weights2: list[np.ndarray] = []
    with torch.no_grad():
        for b_tech, b_macro, b_news, _, _ in test_loader:
            b_tech = b_tech.to(device)
            b_macro = b_macro.to(device)
            b_news = b_news.to(device)

            logits, weights = model(b_tech, b_macro, b_news)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs2.extend(probs)

            if weights is not None:
                all_weights2.append(weights.cpu().numpy())

    test_probs = np.array(all_probs2)
    test_weights = np.vstack(all_weights2) if all_weights2 else None
    # Final visibility: test probability spread detects collapse immediately.
    if len(test_probs):
        logger.info(
            "Test probs: mean=%.4f std=%.4f min=%.4f max=%.4f pos_ratio@0.5=%.3f pos_ratio@opt=%.3f",
            float(np.mean(test_probs)),
            float(np.std(test_probs)),
            float(np.min(test_probs)),
            float(np.max(test_probs)),
            float(np.mean(test_probs >= 0.5)),
            float(np.mean(test_probs >= best_thr)),
        )
        if float(np.std(test_probs)) < 0.01:
            logger.warning(
                "MODEL COLLAPSE DETECTED: test prob std=%.5f — predictions are near-constant!",
                float(np.std(test_probs)),
            )
    return model, test_probs, test_weights, best_thr, history


def evaluate_model_walk_forward(
    model_fn: Callable[[], nn.Module],
    model_name: str,
    tensors: MultiSourceTensors,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    embargo: int = 5,
    seed: int = 42,
    use_class_weight: bool = True,
    threshold: float | None = None,
    threshold_metric: str = "mcc",
    entropy_lambda: float = 0.0,
    aux_lambda: float = 0.0,
    early_stop_metric: str = "loss",
    loss_name: str = "bce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> EvaluationResults:
    """Execute expanding window validation and generate thesis evaluation metrics.

    Partitions (with embargo to prevent 5-day-horizon label leakage):
    - Train set: First 70% of chronological data
    - Embargo gap: ``embargo`` samples dropped after train (default 5 = horizon)
    - Validation set: Next 10% (for early stopping & threshold tuning)
    - Embargo gap: ``embargo`` samples dropped after val
    - Test set: Final 20% of chronological data (unseen future)

    The decision threshold is optimized on VALIDATION (MCC) unless explicitly given.
    """
    set_all_seeds(seed)
    n_samples = len(tensors.y)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Embargo: drop `embargo` samples at each boundary (label horizon overlap).
    train_cut = max(train_end - 0, 0)
    val_start = min(train_end + embargo, n_samples)
    val_cut = val_end
    test_start = min(val_end + embargo, n_samples)

    if test_start >= n_samples:
        raise ValueError(
            f"Embargo={embargo} leaves empty test set (n={n_samples}, val_end={val_end}). "
            "Reduce embargo or adjust ratios."
        )

    logger.info(
        "Split sizes for %s: Train=%d, Val=[%d:%d]=%d, Test=[%d:]=%d (Total=%d, embargo=%d)",
        model_name,
        train_cut,
        val_start,
        val_cut,
        max(val_cut - val_start, 0),
        test_start,
        n_samples - test_start,
        n_samples,
        embargo,
    )

    train_ds = MultiSourceTorchDataset(
        tensors.X_tech[:train_cut],
        tensors.X_macro[:train_cut],
        tensors.X_news[:train_cut],
        tensors.y[:train_cut],
        tensors.returns_5d[:train_cut],
    )
    val_ds = MultiSourceTorchDataset(
        tensors.X_tech[val_start:val_cut],
        tensors.X_macro[val_start:val_cut],
        tensors.X_news[val_start:val_cut],
        tensors.y[val_start:val_cut],
        tensors.returns_5d[val_start:val_cut],
    )
    test_ds = MultiSourceTorchDataset(
        tensors.X_tech[test_start:],
        tensors.X_macro[test_start:],
        tensors.X_news[test_start:],
        tensors.y[test_start:],
        tensors.returns_5d[test_start:],
    )

    model = model_fn()
    _, test_probs, test_weights, val_best_thr, history = train_single_split(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        seed=seed,
        use_class_weight=use_class_weight,
        entropy_lambda=entropy_lambda,
        aux_lambda=aux_lambda,
        early_stop_metric=early_stop_metric,
        loss_name=loss_name,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
    )

    # Threshold: explicit override wins; otherwise use val-optimized threshold.
    best_thr = float(threshold) if threshold is not None else float(val_best_thr)

    test_actuals = tensors.y[test_start:]
    test_returns = tensors.returns_5d[test_start:]
    test_dates = tensors.dates[test_start:]

    # Compute Classification Metrics (Thesis Section 2.7) at val-optimized threshold.
    clf_metrics = compute_classification_metrics(test_actuals, test_probs, threshold=best_thr)

    # Compute Financial / Trading Simulation Metrics (Thesis Section 2.7)
    fin_metrics = simulate_trading_strategy(
        y_prob=test_probs,
        returns_5d=test_returns,
        dates=test_dates,
        threshold=best_thr,
    )

    return EvaluationResults(
        model_name=model_name,
        classification_metrics=clf_metrics,
        financial_metrics=fin_metrics,
        predictions=test_probs,
        actuals=test_actuals,
        test_dates=test_dates,
        gating_weights=test_weights,
        best_threshold=best_thr,
        history=history,
    )
