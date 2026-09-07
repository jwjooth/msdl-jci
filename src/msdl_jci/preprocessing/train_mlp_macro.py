import os
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Fallback-safe MinMaxScaler & Evaluation Metrics
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
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
        return np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / (np.asarray(y_true) + 1e-9)))


MONTH_MAP = {
    'januari': '01', 'jan': '01',
    'februari': '02', 'feb': '02',
    'maret': '03', 'mar': '03',
    'april': '04', 'apr': '04',
    'mei': '05', 'may': '05',
    'juni': '06', 'jun': '06',
    'juli': '07', 'jul': '07',
    'agustus': '08', 'agu': '08', 'aug': '08',
    'september': '09', 'sep': '09',
    'oktober': '10', 'okt': '10', 'oct': '10',
    'november': '11', 'nov': '11',
    'desember': '12', 'des': '12', 'dec': '12'
}


# ==============================================================================
# 1. MACROECONOMIC DATA EXTRACTION & TEMPORAL AS-OF ALIGNMENT
# ==============================================================================
def load_and_align_macro_data(db_path: str = "./berita_ihsg_enterprise.db") -> pd.DataFrame:
    """
    Extracts bi_rate, inflation_data, and kurs_usdidr from SQLite, parses Indonesian
    date formats, and performs point-in-time forward-fill alignment against trading days.
    """
    with sqlite3.connect(db_path) as conn:
        # 1. Master calendar & target from JCI
        df_jci = pd.read_sql_query("SELECT Date, Close AS jci_close FROM jci_historical ORDER BY Date ASC", conn)
        df_jci['Date'] = pd.to_datetime(df_jci['Date'])

        # 2. Daily USD/IDR exchange rate
        df_kurs = pd.read_sql_query("SELECT Date, Close AS usd_idr FROM kurs_usdidr ORDER BY Date ASC", conn)
        df_kurs['Date'] = pd.to_datetime(df_kurs['Date'])

        # 3. Policy Interest Rate (BI 7-Day Reverse Repo Rate)
        df_bi = pd.read_sql_query("SELECT Period, [BI-7Day-RR] AS bi_raw FROM bi_rate", conn)
        ext_bi = df_bi['Period'].str.extract(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})')
        ext_bi[0] = ext_bi[0].str.zfill(2)
        ext_bi[1] = ext_bi[1].str.lower().map(MONTH_MAP)
        df_bi['Date'] = pd.to_datetime(ext_bi[2] + '-' + ext_bi[1] + '-' + ext_bi[0])
        df_bi['bi_rate'] = df_bi['bi_raw'].astype(str).str.replace('%', '', regex=False).str.strip().astype(float)
        df_bi = df_bi[['Date', 'bi_rate']].sort_values('Date').dropna()

        # 4. National Headline Inflation (BPS)
        df_inf = pd.read_sql_query("SELECT Periode, [Data Inflasi] AS inf_raw FROM inflation_data", conn)
        ext_inf = df_inf['Periode'].str.extract(r'([A-Za-z]+)\s+(\d{4})')
        ext_inf[0] = ext_inf[0].str.lower().map(MONTH_MAP)
        df_inf['Date'] = pd.to_datetime(ext_inf[1] + '-' + ext_inf[0] + '-01')
        df_inf['inflation_rate'] = df_inf['inf_raw'].astype(str).str.replace('%', '', regex=False).str.strip().astype(float)
        df_inf = df_inf[['Date', 'inflation_rate']].sort_values('Date').dropna()

    # Temporal As-Of Join (backward merge prevents lookahead bias)
    merged = pd.merge(df_jci, df_kurs, on='Date', how='left')
    merged = pd.merge_asof(merged.sort_values('Date'), df_bi.sort_values('Date'), on='Date', direction='backward')
    merged = pd.merge_asof(merged.sort_values('Date'), df_inf.sort_values('Date'), on='Date', direction='backward')

    # Forward-fill periodic policy values, backfill initial boundary days
    merged['usd_idr'] = merged['usd_idr'].ffill().bfill()
    merged['bi_rate'] = merged['bi_rate'].ffill().bfill()
    merged['inflation_rate'] = merged['inflation_rate'].ffill().bfill()

    # Feature Engineering: Levels and First Differences (Macro Shocks / Momentum)
    merged['usd_idr_pct_change'] = merged['usd_idr'].pct_change().fillna(0.0)
    merged['bi_rate_diff'] = merged['bi_rate'].diff().fillna(0.0)
    merged['inflation_diff'] = merged['inflation_rate'].diff().fillna(0.0)

    print(f"[INFO] Successfully aligned {len(merged)} trading days with macroeconomic variables.")
    return merged


# ==============================================================================
# 2. DATASET LOADER
# ==============================================================================
class MacroDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# 3. MACROECONOMIC MLP ARCHITECTURE
# ==============================================================================
class MacroMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout
    designed to extract non-linear macroeconomic latent representations.
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        # Layer 1: Feature expansion + Normalization
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Layer 2: Representation Compression
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )
        # Layer 3: Latent Bottleneck Embedding (for Soft Gating Fusion in Phase 4)
        self.latent_layer = nn.Sequential(
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU()
        )
        # Regression Prediction Head
        self.regressor = nn.Linear(latent_dim, 1)

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 16-dimensional latent representation vector."""
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        latent = self.latent_layer(h2)
        return latent

    def forward(self, x: torch.Tensor):
        latent = self.extract_latent(x)
        out = self.regressor(latent)
        return out


# ==============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# ==============================================================================
def train_and_evaluate_macro_mlp(
    db_path: str = "./berita_ihsg_enterprise.db",
    train_split: float = 0.8,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    # 1. Load and align data
    df_macro = load_and_align_macro_data(db_path)
    macro_cols = ['usd_idr', 'bi_rate', 'inflation_rate', 'usd_idr_pct_change', 'bi_rate_diff', 'inflation_diff']
    
    X_raw = df_macro[macro_cols].values
    y_raw = df_macro[['jci_close']].values

    # 2. Chronological Split (Identical 80/20 partition as Technical LSTM)
    split_idx = int(len(df_macro) * train_split)
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

    # 3. Scaling strictly on Train set (Zero Data Leakage)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = feature_scaler.fit_transform(X_train_raw)
    X_test = feature_scaler.transform(X_test_raw)

    y_train = target_scaler.fit_transform(y_train_raw).flatten()
    y_test = target_scaler.transform(y_test_raw).flatten()

    # 4. DataLoaders
    train_loader = DataLoader(MacroDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MacroDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 5. Model Initialization
    input_dim = len(macro_cols)
    model = MacroMLP(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32, latent_dim=16, dropout=0.2).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 6. Training with Early Stopping
    best_loss = float('inf')
    early_stop_counter = 0
    best_weights_path = "best_macro_mlp.pt"

    print("\n" + "="*50 + "\n[INFO] Starting Macroeconomic MLP Training...\n" + "="*50)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)

        train_loss /= len(X_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(X_b)

        val_loss /= len(X_test)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.6f} | Test Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch} (Best Val Loss: {best_loss:.6f})")
                break

    # 7. Evaluation & Metric Computation
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        test_preds_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        # Extract full dataset latent representations for Phase 4 fusion
        full_X_scaled = np.vstack([X_train, X_test])
        full_latent = model.extract_latent(torch.tensor(full_X_scaled, dtype=torch.float32).to(device)).cpu().numpy()

    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(test_preds_scaled).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    dir_acc = np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_true[:-1])) * 100

    print("\n" + "="*50)
    print("      FINAL MACROECONOMIC MLP EVALUATION       ")
    print("="*50)
    print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} IDR pts")
    print(f"MAE  (Mean Absolute Error)     : {mae:.2f} IDR pts")
    print(f"MAPE (Mean Absolute % Error)   : {mape:.2f}%")
    print(f"Directional Accuracy (Hit Rate): {dir_acc:.2f}%")
    print("="*50)

    # 8. Save Artifacts for Soft Gating Multi-Modal Fusion
    eval_df = pd.DataFrame({
        'Date': df_macro['Date'].iloc[split_idx:].dt.strftime('%Y-%m-%d').values,
        'Actual_Close': y_true,
        'Predicted_Close': y_pred
    })
    eval_df.to_csv("macro_mlp_evaluation.csv", index=False)

    latent_cols = [f"macro_latent_{i}" for i in range(full_latent.shape[1])]
    latent_df = pd.DataFrame(full_latent, columns=latent_cols)
    latent_df.insert(0, "Date", df_macro['Date'].dt.strftime('%Y-%m-%d').values)
    latent_df.to_csv("macro_latent_features.csv", index=False)

    print(f"[SUCCESS] Model weights saved to '{best_weights_path}'")
    print(f"[SUCCESS] Evaluation saved to 'macro_mlp_evaluation.csv'")
    print(f"[SUCCESS] Latent embeddings ({len(latent_df)}x{len(latent_cols)}) saved to 'macro_latent_features.csv'!")

    return model, latent_df


if __name__ == "__main__":
    train_and_evaluate_macro_mlp()
