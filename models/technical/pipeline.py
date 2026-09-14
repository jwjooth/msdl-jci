"""Technical price prediction pipeline."""

import numpy as np
from numpy.typing import NDArray

from models.technical.technical_encoder import TechnicalDataEncoder


class TechnicalPipeline:
    def __init__(self, look_back: int = 30) -> None:
        self.encoder = TechnicalDataEncoder(look_back=look_back)

    def build_price_prediction(self, price_csv: str) -> NDArray[np.float64]:
        return self.encoder.fit_and_predict(price_csv)
