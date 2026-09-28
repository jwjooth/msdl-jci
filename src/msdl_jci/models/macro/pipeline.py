"""Macro-economic pipeline orchestration."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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

        Omitted sources fall back to the SQLite tables.
        """
        return self.encoder.fit_and_encode(
            bi_rate_csv=bi_rate_csv,
            inflation_csv=inflation_csv,
            kurs_csv=kurs_csv,
        )
