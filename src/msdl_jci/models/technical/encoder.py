"""Technical price encoder based on scikit-learn regressor."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from msdl_jci.config.settings import get_settings
from msdl_jci.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedPriceResult:
    model: MLPRegressor
    scaler: MinMaxScaler
    predictions: NDArray[np.float64]
    look_back: int


class TechnicalDataEncoder:
    """Encoder for historical financial price series."""

    def __init__(self, look_back: int = 30) -> None:
        self.look_back = look_back
        self.price_result: TrainedPriceResult | None = None

    def fit_and_predict(
        self,
        csv_path: str | Path,
        column_name: str = "Close",
    ) -> NDArray[np.float64]:
        """Train model on historical price series and return predictions."""
        logger.info("Training JCI Historical Price model from %s...", csv_path)
        self.price_result = self._train_price_series(csv_path, column_name=column_name)
        return self.price_result.predictions

    def _train_price_series(
        self,
        csv_path: str | Path,
        column_name: str = "Close",
    ) -> TrainedPriceResult:
        path_obj = Path(csv_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found at: {path_obj}")

        df = pd.read_csv(path_obj)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", column_name]).sort_values("Date")
            df = df.drop_duplicates(subset=["Date"], keep="last")
        else:
            df = df.dropna(subset=[column_name])

        values = pd.to_numeric(df[column_name], errors="coerce").dropna().to_numpy(dtype=np.float64)
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
        settings = get_settings()
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=settings.ML_MAX_ITER,
            random_state=settings.ML_RANDOM_STATE,
        )
        model.fit(x_arr, y_arr)

        prediction = model.predict(x_arr).reshape(-1, 1)

        return TrainedPriceResult(
            model=model,
            scaler=scaler,
            predictions=prediction,
            look_back=self.look_back,
        )
