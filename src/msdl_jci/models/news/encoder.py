"""Financial news encoder and representation model."""

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
class TrainedNewsResult:
    model: MLPRegressor
    scaler: MinMaxScaler
    prediction: NDArray[np.float64]
    look_back: int


class NewsDataEncoder:
    """Encoder for financial news embeddings or aggregated sentiment signals."""

    def __init__(self, look_back: int = 30) -> None:
        self.look_back = look_back
        self.news_result: TrainedNewsResult | None = None

    def fit_and_predict(
        self,
        csv_path: str | Path,
        feature_column: str = "mean_embedding",
    ) -> NDArray[np.float64]:
        """Fit model on news feature representation and predict sequence."""
        logger.info("Training Financial News model from %s...", csv_path)
        path_obj = Path(csv_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"News dataset not found at: {path_obj}")

        df = pd.read_csv(path_obj)
        if feature_column not in df.columns:
            # Fallback: take first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise KeyError(f"No numeric feature column found in {path_obj}")
            feature_column = numeric_cols[0]

        values = pd.to_numeric(df[feature_column], errors="coerce").dropna().to_numpy(dtype=np.float64)
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

        self.news_result = TrainedNewsResult(
            model=model,
            scaler=scaler,
            prediction=prediction,
            look_back=self.look_back,
        )
        return prediction
