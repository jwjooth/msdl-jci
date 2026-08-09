"""Standalone MLP trainer for Inflation rate data."""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def train_inflation_model(
    csv_path: str, look_back: int = 3
) -> tuple[MLPRegressor, MinMaxScaler]:
    """Train an MLP regressor on Inflation time series.

    Args:
        csv_path: Path to the Inflation CSV file.
        look_back: Number of past periods for the sliding window.

    Returns:
        Tuple of (fitted MLPRegressor, fitted MinMaxScaler).
    """
    df = pd.read_csv(csv_path)
    df["Data Inflasi"] = (
        df["Data Inflasi"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(float)
    )

    data = df["Data Inflasi"].values.reshape(-1, 1)
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
