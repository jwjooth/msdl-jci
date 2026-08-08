import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def create_dataset(data_set, look_back):
    X, y = [], []

    for i in range(len(data_set) - look_back):
        X.append(data_set[i:i + look_back, 0])
        y.append(data_set[i + look_back, 0])

    return np.array(X), np.array(y)


def predict_bi_rate(data):
    df = data.copy()

    # =========================
    # Data cleaning
    # =========================

    df["BI-7Day-RR"] = (
        df["BI-7Day-RR"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(float)
    )

    values = df["BI-7Day-RR"].values.reshape(-1, 1)

    # =========================
    # Configuration
    # =========================

    look_back = 3

    # =========================
    # Chronological split
    # =========================

    split_index = int(len(values) * 0.8)

    train_raw = values[:split_index]

    # Keep previous look_back observations
    # as context for the first test sequence.
    test_raw = values[split_index - look_back:]

    # =========================
    # Scaling
    # =========================

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    # =========================
    # Create sequences
    # =========================

    X_train, y_train = create_dataset(
        train_scaled,
        look_back,
    )

    X_test, y_test = create_dataset(
        test_scaled,
        look_back,
    )

    # =========================
    # MLP
    # =========================

    model = MLPRegressor(
        hidden_layer_sizes=(100, 100),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # =========================
    # Prediction
    # =========================

    prediction_scaled = model.predict(X_test)

    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    # =========================
    # Evaluation
    # =========================

    mse = mean_squared_error(actual, prediction)
    mae = mean_absolute_error(actual, prediction)
    mape = mean_absolute_percentage_error(actual, prediction)

    return {
        "prediction": prediction,
        "actual": actual,
        "metrics": {
            "mse": mse,
            "mae": mae,
            "mape": mape,
        },
    }