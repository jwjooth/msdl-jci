"""Macro-economic data encoder."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from config.config import get_settings
from utils.data_loader import read_numeric_series
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedMLPResult:
    model: MLPRegressor
    scaler: MinMaxScaler
    predictions: NDArray[np.float64]


class MacroDataEncoder:
    def __init__(self, look_back: int = 3) -> None:
        self.look_back = look_back
        self.bi_result: TrainedMLPResult | None = None
        self.inflation_result: TrainedMLPResult | None = None
        self.kurs_result: TrainedMLPResult | None = None
        self._encoded_features: NDArray[np.float64] | None = None

    def _train_single_series(self, csv_path: str, column_name: str, remove_symbol: str = "") -> TrainedMLPResult:
        values = read_numeric_series(csv_path, column_name, remove_symbol)
        data = values.reshape(-1, 1)

        if len(data) <= self.look_back:
            raise ValueError("Not enough rows for the configured look_back.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        x, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            x.append(scaled_data[i : i + self.look_back, 0])
            y.append(scaled_data[i + self.look_back, 0])

        x_arr, y_arr = np.array(x), np.array(y)
        settings = get_settings()
        model = MLPRegressor(
            hidden_layer_sizes=settings.ML_HIDDEN_LAYERS,
            max_iter=settings.ML_MAX_ITER,
            random_state=settings.ML_RANDOM_STATE,
        )
        model.fit(x_arr, y_arr)

        predictions = model.predict(x_arr).reshape(-1, 1)
        return TrainedMLPResult(model=model, scaler=scaler, predictions=predictions)

    def fit_and_encode(self, bi_rate_csv: str, inflation_csv: str, kurs_csv: str):
        logger.info("Training BI Rate model...")
        self.bi_result = self._train_single_series(bi_rate_csv, "BI-7Day-RR", "%")
        logger.info("Training Inflation model...")
        self.inflation_result = self._train_single_series(inflation_csv, "Data Inflasi", "%")
        logger.info("Training Kurs USD/IDR model...")
        self.kurs_result = self._train_single_series(kurs_csv, "Close")

        min_len = min(
            len(self.bi_result.predictions),
            len(self.inflation_result.predictions),
            len(self.kurs_result.predictions),
        )

        self._encoded_features = np.column_stack(
            (
                self.bi_result.predictions[:min_len],
                self.inflation_result.predictions[:min_len],
                self.kurs_result.predictions[:min_len],
            )
        )
        return self._encoded_features

    def get_encoded_features(self) -> NDArray[np.float64]:
        if self._encoded_features is None:
            raise RuntimeError("Call fit_and_encode() first.")
        return self._encoded_features
