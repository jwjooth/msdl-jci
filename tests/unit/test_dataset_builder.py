<<<<<<< HEAD
"""Unit tests for MultiSourceDatasetBuilder."""

import numpy as np
import pandas as pd

from msdl_jci.utils.dataset_builder import (
    MultiSourceDatasetBuilder,
    calculate_technical_indicators,
    parse_indonesian_date,
)


def test_parse_indonesian_date():
    dates = pd.Series(["17 Desember 2025", "19 November 2024", "01 Januari 2023"])
    parsed = parse_indonesian_date(dates)
    assert parsed.iloc[0] == pd.Timestamp("2025-12-17")
    assert parsed.iloc[1] == pd.Timestamp("2024-11-19")
    assert parsed.iloc[2] == pd.Timestamp("2023-01-01")


def test_calculate_technical_indicators():
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = np.linspace(100, 150, n) + np.random.normal(0, 1, n)
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": prices + 2,
        "Low": prices - 2,
        "Close": prices,
        "Volume": np.full(n, 1000000.0),
    })

    feat_df = calculate_technical_indicators(df)
    assert "RSI_14" in feat_df.columns
    assert "MACD" in feat_df.columns
    assert "MACD_Signal" in feat_df.columns
    assert "ATR_14" in feat_df.columns
    assert "SMA_20" in feat_df.columns

    # Verify RSI values fall within [0, 100]
    valid_rsi = feat_df["RSI_14"].dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()


def test_dataset_builder_tensors():
    builder = MultiSourceDatasetBuilder(look_back=10, prediction_horizon=3)
    df = builder.build_aligned_dataframe()
    assert len(df) > 500
    assert "target_direction" in df.columns
    assert "return_5d" in df.columns

    tensors, f_scaler, m_scaler = builder.create_multisource_tensors(df)
    assert tensors.X_tech.ndim == 3
    assert tensors.X_tech.shape[1] == 10  # look_back
    assert tensors.X_tech.shape[2] == len(builder.tech_feature_cols)
    assert tensors.X_macro.ndim == 2
    assert tensors.X_macro.shape[1] == 3
    assert tensors.X_news.ndim == 2
    assert tensors.X_news.shape[1] == 768
    assert len(tensors.y) == len(tensors.X_tech)
||||||| c9f8f7e
=======
"""Unit tests for MultiSourceDatasetBuilder."""

import numpy as np
import pandas as pd
import pytest
from msdl_jci.utils.dataset_builder import (
    MultiSourceDatasetBuilder,
    calculate_technical_indicators,
    parse_indonesian_date,
)


def test_parse_indonesian_date():
    dates = pd.Series(["17 Desember 2025", "19 November 2024", "01 Januari 2023"])
    parsed = parse_indonesian_date(dates)
    assert parsed.iloc[0] == pd.Timestamp("2025-12-17")
    assert parsed.iloc[1] == pd.Timestamp("2024-11-19")
    assert parsed.iloc[2] == pd.Timestamp("2023-01-01")


def test_calculate_technical_indicators():
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = np.linspace(100, 150, n) + np.random.normal(0, 1, n)
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": prices + 2,
        "Low": prices - 2,
        "Close": prices,
        "Volume": np.full(n, 1000000.0),
    })

    feat_df = calculate_technical_indicators(df)
    assert "RSI_14" in feat_df.columns
    assert "MACD" in feat_df.columns
    assert "MACD_Signal" in feat_df.columns
    assert "ATR_14" in feat_df.columns
    assert "SMA_20" in feat_df.columns

    # Verify RSI values fall within [0, 100]
    valid_rsi = feat_df["RSI_14"].dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()


def test_dataset_builder_tensors():
    builder = MultiSourceDatasetBuilder(look_back=10, prediction_horizon=3)
    df = builder.build_aligned_dataframe()
    assert len(df) > 500
    assert "target_direction" in df.columns
    assert "return_5d" in df.columns

    tensors, f_scaler, m_scaler = builder.create_multisource_tensors(df)
    assert tensors.X_tech.ndim == 3
    assert tensors.X_tech.shape[1] == 10  # look_back
    assert tensors.X_tech.shape[2] == len(builder.tech_feature_cols)
    assert tensors.X_macro.ndim == 2
    assert tensors.X_macro.shape[1] == 3
    assert tensors.X_news.ndim == 2
    assert tensors.X_news.shape[1] == 768
    assert len(tensors.y) == len(tensors.X_tech)
>>>>>>> main
