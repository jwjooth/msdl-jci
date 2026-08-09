"""Service layer for macro feature generation."""

from numpy.typing import NDArray
import numpy as np

from msdl_jci.domain.macro.encoder import MacroDataEncoder


class MacroEncodingService:
    def __init__(self, look_back: int = 3) -> None:
        self.encoder = MacroDataEncoder(look_back=look_back)

    def build_macro_embeddings(
        self,
        bi_rate_csv: str,
        inflation_csv: str,
        kurs_csv: str,
    ) -> NDArray[np.float64]:
        return self.encoder.fit_and_encode(bi_rate_csv, inflation_csv, kurs_csv)
