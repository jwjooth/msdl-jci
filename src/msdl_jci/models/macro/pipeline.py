"""Macro-economic pipeline orchestration."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from msdl_jci.config.settings import get_settings
from msdl_jci.models.macro.encoder import MacroDataEncoder


class MacroPipeline:
    """Orchestration pipeline for macro economic data encoding."""

    def __init__(self, look_back: int | None = None) -> None:
        self.encoder = MacroDataEncoder(look_back=look_back)

    def build_macro_embeddings(
        self,
        bi_rate_csv: str | Path | None = None,
        inflation_csv: str | Path | None = None,
        kurs_csv: str | Path | None = None,
    ) -> NDArray[np.float64] | None:
        """Train models and generate combined macro feature embeddings.

        Falls back to configured default paths in Settings if not explicitly provided.
        """
        settings = get_settings()
        bi_path = bi_rate_csv if bi_rate_csv is not None else settings.BI_RATE_CSV
        inflation_path = (
            inflation_csv if inflation_csv is not None else settings.INFLATION_CSV
        )
        kurs_path = kurs_csv if kurs_csv is not None else settings.KURS_CSV

        return self.encoder.fit_and_encode(
            bi_rate_csv=bi_path,
            inflation_csv=inflation_path,
            kurs_csv=kurs_path,
        )
