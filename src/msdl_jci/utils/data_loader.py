"""CSV and data loading helpers."""

from pathlib import Path
from typing import Union
import numpy as np
from numpy.typing import NDArray
import pandas as pd


def read_numeric_series(
    csv_path: Union[str, Path],
    column_name: str,
    remove_symbol: str = "",
) -> NDArray[np.float64]:
    """Read a numeric series from a CSV file with automatic symbol stripping."""
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found at path: {path_obj}")

    df = pd.read_csv(path_obj)

    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found. Available: {list(df.columns)}"
        )

    if remove_symbol:
        cleaned = (
            df[column_name]
            .astype(str)
            .str.replace(remove_symbol, "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="raise").to_numpy(dtype=np.float64)

    return pd.to_numeric(df[column_name], errors="raise").to_numpy(dtype=np.float64)
