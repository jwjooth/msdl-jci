"""CSV loading helpers."""

import pandas as pd


def read_numeric_series(csv_path: str, column_name: str, remove_symbol: str = ""):
    df = pd.read_csv(csv_path)

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
        return pd.to_numeric(cleaned, errors="raise").to_numpy()

    return pd.to_numeric(df[column_name], errors="raise").to_numpy()
