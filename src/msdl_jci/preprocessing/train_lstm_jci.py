import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Safe imports for Scaler and Metrics with native pure-NumPy fallbacks
try:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )
    from sklearn.preprocessing import MinMaxScaler
except ImportError:

    class MinMaxScaler:
        def __init__(self, feature_range=(0, 1)):
            self.min_val = None
            self.max_val = None
            self.feature_range = feature_range

        def fit_transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.min_val = np.nanmin(X, axis=0)
            self.max_val = np.nanmax(X, axis=0)
            range_val = np.where(self.max_val - self.min_val == 0, 1.0, self.max_val - self.min_val)
            scaled = (X - self.min_val) / range_val
            return scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        def transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            range_val = np.where(self.max_val - self.min_val == 0, 1.0, self.max_val - self.min_val)
            scaled = (X - self.min_val) / range_val
            return scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        def inverse_transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            range_val = np.where(self.max_val - self.min_val == 0, 1.0, self.max_val - self.min_val)
            unscaled = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            return unscaled * range_val + self.min_val

    def mean_squared_error(y_true, y_pred):
        return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(
            np.abs((np.asarray(y_true) - np.asarray(y_pred)) / (np.asarray(y_true) + 1e-9))
        )


# ==============================================================================
# 1. FEATURE ENGINEERING (TECHNICAL INDICATORS)
# ==============================================================================
def calculate_technical_indicators(
    df: pd.DataFrame, rsi_period: int = 14, atr_period: int = 14
) -> pd.DataFrame:
    """
    Computes technical momentum and volatility indicators (RSI & ATR) from OHLCV data.

    Mathematical Foundations:
    1. RSI (Relative Strength Index - Wilder's Smoothing):
       - Measures the speed and change of price movements (momentum oscillator: 0 - 100).
       - Delta = Close_t - Close_{t-1}
       - RS = EMA(Gain, period) / EMA(Loss, period)
       - RSI = 100 - (100 / (1 + RS))

    2. ATR (Average True Range):
       - Quantifies market volatility by decomposing the entire range of an asset.
       - TR = max(High - Low, |High - Close_{prev}|, |Low - Close_{prev}|)
       - ATR = EMA(TR, period)
    """
    df = df.copy()

    # Ensure numerical types and sorting
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # --- 1.1 Calculate RSI ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Exponential Weighted Moving Average (Wilder's style: alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- 1.2 Calculate ATR ---
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period, adjust=False).mean()

    # --- 1.3 Moving Average Context (SMA-20) ---
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Drop warm-up NaN rows caused by indicator rolling windows
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


# ==============================================================================
# 2. DATASET & SLIDING WINDOW GENERATOR
# ==============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sliding_windows(features: np.ndarray, target: np.ndarray, lookback: int = 30):
    """
    Transforms 2D tabular features into 3D sequential sliding windows for LSTM.
    Shape: [Samples, Lookback, Num_Features] -> Target: [Samples, 1]
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback : i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ==============================================================================
# 3. LSTM NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class JCIStockLSTM(nn.Module):
    """
    Stacked LSTM with Dropout regularization and dense regression head for IHSG price prediction.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take hidden state of the last time-step
        last_step_out = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_dim]
        out = self.regressor(last_step_out)
        return out


# ==============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# ==============================================================================
def train_and_evaluate_jci_lstm(
    db_path: str = "./berita_ihsg_enterprise.db",
    lookback: int = 30,
    train_split: float = 0.8,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on compute device: {device}")

    # --- Step 1: Load Data from SQLite ---
    with sqlite3.connect(db_path) as conn:
        df_raw = pd.read_sql_query("SELECT * FROM jci_historical ORDER BY Date ASC", conn)
    print(f"[INFO] Loaded {len(df_raw)} historical records from 'jci_historical'")

    # --- Step 2: Feature Engineering (OHLCV + RSI + ATR + SMA) ---
    df_feat = calculate_technical_indicators(df_raw, rsi_period=14, atr_period=14)
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "ATR", "SMA_20"]
    print(f"[INFO] Features computed: {feature_cols}")

    data_values = df_feat[feature_cols].values
    close_values = df_feat[["Close"]].values

    # --- Step 3: Chronological Train-Test Split (Strict Time-Series Order) ---
    split_idx = int(len(data_values) * train_split)
    train_data = data_values[:split_idx]
    test_data = data_values[split_idx:]

    train_target = close_values[:split_idx]
    test_target = close_values[split_idx:]

    # --- Step 4: Scaling without Data Leakage ---
    # Fit scaler strictly on training distribution only
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = feature_scaler.fit_transform(train_data)
    test_scaled = feature_scaler.transform(test_data)

    train_target_scaled = target_scaler.fit_transform(train_target).flatten()
    test_target_scaled = target_scaler.transform(test_target).flatten()

    # --- Step 5: Sliding Window 3D Tensor Construction ---
    X_train, y_train = create_sliding_windows(train_scaled, train_target_scaled, lookback=lookback)

    # For test set, append last 'lookback' steps of train to prevent lookahead boundary loss
    full_test_features = np.vstack([train_scaled[-lookback:], test_scaled])
    full_test_target = np.concatenate([train_target_scaled[-lookback:], test_target_scaled])
    X_test, y_test = create_sliding_windows(full_test_features, full_test_target, lookback=lookback)

    print(f"[INFO] Tensor Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Create PyTorch DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Step 6: Initialize Model, Loss, Optimizer & Scheduler ---
    input_dim = len(feature_cols)
    model = JCIStockLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2).to(device)

    criterion = nn.HuberLoss(delta=1.0)  # Robust against extreme stock outlier shocks
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- Step 7: Training Loop with Early Stopping ---
    best_loss = float("inf")
    early_stop_counter = 0
    best_weights_path = "best_jci_lstm.pt"

    print("\n" + "=" * 50 + "\n[INFO] Starting LSTM Model Training...\n" + "=" * 50)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)

        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(X_b)

        val_loss /= len(test_dataset)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.6f} | Test Loss: {val_loss:.6f}"
            )

        # Check early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(
                    f"[INFO] Early stopping triggered at epoch {epoch} (Best Val Loss: {best_loss:.6f})"
                )
                break

    # --- Step 8: Final Model Evaluation & Metric Computation ---
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        test_preds_scaled = (
            model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        )

    # Inverse transform to original IDR price scale
    y_true_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = target_scaler.inverse_transform(test_preds_scaled).flatten()

    # Quantitative Financial Metrics
    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_true_actual, y_pred_actual) * 100

    # Directional Accuracy / Hit Rate (% of correct price direction movements)
    dir_true = np.sign(y_true_actual[1:] - y_true_actual[:-1])
    dir_pred = np.sign(y_pred_actual[1:] - y_true_actual[:-1])
    directional_accuracy = np.mean(dir_true == dir_pred) * 100

    print("\n" + "=" * 50)
    print("       FINAL JCI / IHSG EVALUATION METRICS       ")
    print("=" * 50)
    print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} IDR pts")
    print(f"MAE  (Mean Absolute Error)     : {mae:.2f} IDR pts")
    print(f"MAPE (Mean Absolute % Error)   : {mape:.2f}%")
    print(f"Directional Accuracy (Hit Rate): {directional_accuracy:.2f}%")
    print("=" * 50)

    # Save evaluation predictions
    test_dates = df_feat["Date"].iloc[split_idx:].values
    eval_df = pd.DataFrame(
        {"Date": test_dates, "Actual_Close": y_true_actual, "Predicted_Close": y_pred_actual}
    )
    eval_df.to_csv("jci_lstm_evaluation.csv", index=False)
    print("[SUCCESS] Evaluation results saved to 'jci_lstm_evaluation.csv'!")

    return model, eval_df


if __name__ == "__main__":
    train_and_evaluate_jci_lstm(
        db_path="./berita_ihsg_enterprise.db",
        lookback=30,
        train_split=0.8,
        batch_size=32,
        epochs=80,
        lr=1e-3,
        patience=12,
    )
