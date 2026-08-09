"""Technical price encoder based on a lightweight scikit-learn regressor."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedPriceResult:
    model: MLPRegressor
    scaler: MinMaxScaler
    predictions: NDArray[np.float64]
    look_back: int


class TechnicalDataEncoder:
    def __init__(self, look_back: int = 30) -> None:
        self.look_back = look_back
        self.price_result: TrainedPriceResult | None = None

    def fit_and_predict(self, csv_path: str) -> NDArray[np.float64]:
        logger.info("Training JCI Historical Price model...")
        self.price_result = self._train_price_series(csv_path)
        return self.price_result.predictions

    def _train_price_series(
        self,
        csv_path: str,
        column_name: str = "Close",
    ) -> TrainedPriceResult:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", column_name]).sort_values("Date")
        df = df.drop_duplicates(subset=["Date"], keep="last")

        values = pd.to_numeric(df[column_name], errors="coerce").dropna().to_numpy()
        if len(values) <= self.look_back:
            raise ValueError(
                f"Data has {len(values)} rows but look_back={self.look_back}. "
                f"Need at least {self.look_back + 1} rows."
            )

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values.reshape(-1, 1))

        x, y = [], []
        for i in range(len(scaled_values) - self.look_back):
            x.append(scaled_values[i : i + self.look_back, 0])
            y.append(scaled_values[i + self.look_back, 0])

        x_arr, y_arr = np.array(x), np.array(y)
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
        )
        model.fit(x_arr, y_arr)

        prediction = model.predict(x_arr).reshape(-1, 1)

        return TrainedPriceResult(
            model=model,
            scaler=scaler,
            predictions=prediction,
            look_back=self.look_back,
        )
