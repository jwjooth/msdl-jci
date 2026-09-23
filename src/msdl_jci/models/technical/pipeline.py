"""Technical price prediction pipeline."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from msdl_jci.config.settings import get_settings
from msdl_jci.models.technical.encoder import TechnicalDataEncoder


class TechnicalPipeline:
    """Orchestration pipeline for financial price prediction."""

    def __init__(self, look_back: int = 30) -> None:
        self.encoder = TechnicalDataEncoder(look_back=look_back)

    def build_price_prediction(
        self,
        price_csv: str | Path | None = None,
        column_name: str = "Close",
    ) -> NDArray[np.float64]:
        """Train model and return predictions, using Settings default path if omitted."""
        settings = get_settings()
        target_path = price_csv if price_csv is not None else settings.JCI_HISTORICAL_CSV
        return self.encoder.fit_and_predict(target_path, column_name=column_name)
