"""Standalone MLP trainer for BI-7Day Reverse Repo Rate."""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def train_bi_rate_model(
    csv_path: str, look_back: int = 3
) -> tuple[MLPRegressor, MinMaxScaler]:
    """Train an MLP regressor on BI-7Day-RR time series.

    Args:
        csv_path: Path to the BI Rate CSV file.
        look_back: Number of past periods for the sliding window.

    Returns:
        Tuple of (fitted MLPRegressor, fitted MinMaxScaler).
    """
    df = pd.read_csv(csv_path)
    df["BI-7Day-RR"] = (
        df["BI-7Day-RR"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(float)
    )

    data = df["BI-7Day-RR"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i : (i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])

    X, y = np.array(X), np.array(y)

    model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    model.fit(X, y)

    return model, scaler
