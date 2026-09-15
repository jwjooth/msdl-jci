"""Unit tests for configuration and settings."""

from msdl_jci.config.settings import Settings, get_settings


def test_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert isinstance(s1, Settings)


def test_settings_paths_exist():
    settings = get_settings()
    assert settings.ROOT_DIR.exists()
    assert settings.DATA_DIR.exists()
    assert settings.DATA_RAW_DIR.exists()
    assert settings.BI_RATE_CSV.exists()
    assert settings.INFLATION_CSV.exists()
    assert settings.KURS_CSV.exists()
    assert settings.JCI_HISTORICAL_CSV.exists()


def test_settings_hyperparameters():
    settings = get_settings()
    assert settings.ML_LOOK_BACK > 0
    assert settings.ML_MAX_ITER > 0
    assert len(settings.ML_HIDDEN_LAYERS) >= 1
