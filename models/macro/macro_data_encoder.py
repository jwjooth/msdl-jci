"""Macro-economic data encoder using MLP autoencoders.

Trains individual MLP models on BI Rate, Inflation, and Kurs USD/IDR
time series, then concatenates their encoded representations into a
single feature vector for downstream fusion with LSTM and IndoBERT.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class TrainedMLPResult:
    """Container for a trained MLP model and its artifacts."""

    model: MLPRegressor
    scaler: MinMaxScaler
    predictions: NDArray[np.float64]


class MacroDataEncoder:
    """Encodes macro-economic time series into feature vectors via MLP.

    Each macro indicator (BI Rate, Inflation, Kurs) is independently
    trained as a time-series regression task using a sliding window
    of ``look_back`` periods. The predicted outputs are concatenated
    into a single macro feature matrix.

    Args:
        look_back: Number of past periods to use as input features.
    """

    def __init__(self, look_back: int = 3) -> None:
        self.look_back = look_back
        self.bi_result: Optional[TrainedMLPResult] = None
        self.inflation_result: Optional[TrainedMLPResult] = None
        self.kurs_result: Optional[TrainedMLPResult] = None
        self._encoded_features: Optional[NDArray[np.float64]] = None

    def _train_single_mlp(
        self,
        csv_path: str,
        column_name: str,
        remove_symbol: str = "",
    ) -> TrainedMLPResult:
        """Train a single MLP regressor on one macro indicator.

        Args:
            csv_path: Path to the CSV file.
            column_name: Column name containing the target values.
            remove_symbol: Optional symbol to strip (e.g., '%').

        Returns:
            TrainedMLPResult with fitted model, scaler, and predictions.

        Raises:
            FileNotFoundError: If csv_path doesn't exist.
            KeyError: If column_name is not found in the CSV.
            ValueError: If data has fewer rows than look_back.
        """
        df = pd.read_csv(csv_path)

        if column_name not in df.columns:
            raise KeyError(
                f"Column '{column_name}' not found. "
                f"Available: {list(df.columns)}"
            )

        if remove_symbol:
            df[column_name] = (
                df[column_name]
                .astype(str)
                .str.replace(remove_symbol, "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
                .astype(float)
            )
        else:
            df[column_name] = pd.to_numeric(df[column_name], errors="raise")

        data = df[column_name].values.reshape(-1, 1)

        if len(data) <= self.look_back:
            raise ValueError(
                f"Data has {len(data)} rows but look_back={self.look_back}. "
                f"Need at least {self.look_back + 1} rows."
            )

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create time series sequences using sliding window
        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i : i + self.look_back, 0])
            y.append(scaled_data[i + self.look_back, 0])

        X_arr, y_arr = np.array(X), np.array(y)

        model = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            max_iter=500,
            random_state=42,
        )
        model.fit(X_arr, y_arr)

        predictions = model.predict(X_arr).reshape(-1, 1)

        return TrainedMLPResult(
            model=model, scaler=scaler, predictions=predictions
        )

    def fit_and_encode(
        self,
        bi_rate_csv: str,
        inflation_data_csv: str,
        kurs_csv: str,
    ) -> NDArray[np.float64]:
        """Train all macro MLP models and produce encoded features.

        Each macro indicator may have a different number of data points.
        The resulting feature matrix is truncated to the shortest series
        so all columns align.

        Args:
            bi_rate_csv: Path to BI Rate CSV.
            inflation_data_csv: Path to Inflation CSV.
            kurs_csv: Path to Kurs USD/IDR CSV.

        Returns:
            Encoded macro feature matrix of shape (n_samples, 3).
        """
        logger.info("Training BI Rate model...")
        self.bi_result = self._train_single_mlp(
            bi_rate_csv, "BI-7Day-RR", remove_symbol="%"
        )

        logger.info("Training Inflation model...")
        self.inflation_result = self._train_single_mlp(
            inflation_data_csv, "Data Inflasi", remove_symbol="%"
        )

        logger.info("Training Kurs USD/IDR model...")
        self.kurs_result = self._train_single_mlp(kurs_csv, "Close")

        # Truncate to the shortest prediction array so all columns align
        min_len = min(
            len(self.bi_result.predictions),
            len(self.inflation_result.predictions),
            len(self.kurs_result.predictions),
        )

        logger.info(
            f"Aligning prediction arrays to shortest length: {min_len} "
            f"(bi={len(self.bi_result.predictions)}, "
            f"inflation={len(self.inflation_result.predictions)}, "
            f"kurs={len(self.kurs_result.predictions)})"
        )

        self._encoded_features = np.column_stack((
            self.bi_result.predictions[:min_len],
            self.inflation_result.predictions[:min_len],
            self.kurs_result.predictions[:min_len],
        ))

        logger.info(
            f"Encoded macro features shape: "
            f"{self._encoded_features.shape}"
        )
        return self._encoded_features

    def get_encoded_features(self) -> NDArray[np.float64]:
        """Return the encoded macro feature matrix.

        Raises:
            RuntimeError: If fit_and_encode() hasn't been called yet.
        """
        if self._encoded_features is None:
            raise RuntimeError(
                "No encoded features available. "
                "Call fit_and_encode() first."
            )
        return self._encoded_features
