from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import dtype, float64, ndarray
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from msdl_jci.utils.data_loader import read_numeric_series
from msdl_jci.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedMLPResult:
    model: MLPRegressor
    scaler: MinMaxScaler
    predictions: NDArray[np.float64]


class MacroDataEncoder:
    def __init__(self, look_back: int | None = None) -> None:
        settings = get_settings()
        self.look_back = look_back if look_back is not None else settings.ML_LOOK_BACK
        self.bi_result: TrainedMLPResult | None = None
        self.inflation_result: TrainedMLPResult | None = None
        self.kurs_result: TrainedMLPResult | None = None
        self._encoded_features: NDArray[np.float64] | None = None

    def _train_single_series(
        self,
        csv_path: str | Path,
        column_name: str,
        remove_symbol: str = "",
    ) -> TrainedMLPResult:
        values = read_numeric_series(csv_path, column_name, remove_symbol)
        data = values.reshape(-1, 1)

        if len(data) <= self.look_back:
            raise ValueError(
                f"Not enough rows ({len(data)}) for look_back={self.look_back}."
            )

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

    def fit_and_encode(
        self,
        bi_rate_csv: str | Path,
        inflation_csv: str | Path,
        kurs_csv: str | Path,
    ) -> ndarray[tuple[Any, ...], dtype[float64]] | None:
        logger.info("Training BI Rate model from %s...", bi_rate_csv)
        self.bi_result = self._train_single_series(bi_rate_csv, "BI-7Day-RR", "%")

        logger.info("Training Inflation model from %s...", inflation_csv)
        self.inflation_result = self._train_single_series(inflation_csv, "Data Inflasi", "%")

        logger.info("Training Kurs USD/IDR model from %s...", kurs_csv)
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
