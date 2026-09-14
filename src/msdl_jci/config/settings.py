"""Application settings and project paths configuration."""

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root: this file is at src/msdl_jci/config/settings.py -> parents[3] is project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load .env candidates safely
for env_candidate in (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "development" / ".env",
):
    if env_candidate.exists():
        load_dotenv(dotenv_path=env_candidate, override=False)
        break


@dataclass(frozen=True)
class Settings:
    PROJECT_NAME: str = "MSDL-JCI"
    VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Directory Paths
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_MACRO_DIR: Path = PROJECT_ROOT / "data" / "raw" / "macro"
    DATA_TECHNICAL_DIR: Path = PROJECT_ROOT / "data" / "raw" / "technical"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    DATA_DATABASE_DIR: Path = PROJECT_ROOT / "data" / "database"

    # Default file paths
    BI_RATE_CSV: Path = PROJECT_ROOT / "data" / "raw" / "macro" / "bi_rate.csv"
    INFLATION_CSV: Path = PROJECT_ROOT / "data" / "raw" / "macro" / "inflation_data.csv"
    KURS_CSV: Path = PROJECT_ROOT / "data" / "raw" / "macro" / "kurs_usdidr.csv"
    JCI_HISTORICAL_CSV: Path = PROJECT_ROOT / "data" / "raw" / "technical" / "jci_historical.csv"
    SQLITE_DB_PATH: Path = PROJECT_ROOT / "data" / "database" / "berita_ihsg_enterprise.db"
    DAILY_NEWS_EMBEDDINGS_CSV: Path = PROJECT_ROOT / "data" / "processed" / "daily_news_embeddings.csv"
    PROCESSED_DAILY_NEWS_CSV: Path = PROJECT_ROOT / "data" / "processed" / "processed_daily_news.csv"

    # Database Configuration (PostgreSQL / Relational)
    DB_NAME: str = os.getenv("DB_NAME", os.getenv("db_name", "msdl-jci"))
    DB_USER: str = os.getenv("DB_USER", os.getenv("db_user", "postgres"))
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", os.getenv("db_password", "postgres"))
    DB_HOST: str = os.getenv("DB_HOST", os.getenv("db_host", "localhost"))
    DB_PORT: str = os.getenv("DB_PORT", os.getenv("db_port", "5432"))

    # Machine Learning / Modeling Hyperparameters (Aligned with Thesis Table 5)
    ML_LOOK_BACK: int = int(os.getenv("ML_LOOK_BACK", "28"))
    ML_PREDICTION_HORIZON: int = int(os.getenv("ML_PREDICTION_HORIZON", "5"))
    ML_HIDDEN_LAYERS: tuple[int, ...] = (50, 25)
    ML_MAX_ITER: int = int(os.getenv("ML_MAX_ITER", "500"))
    ML_RANDOM_STATE: int = int(os.getenv("ML_RANDOM_STATE", "42"))

    # Multi-Source Architecture Specifications (Thesis Table 5)
    LSTM_HIDDEN_DIM: int = 64
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.2
    MACRO_INPUT_DIM: int = 3
    MACRO_HIDDEN_DIMS: tuple[int, ...] = (32, 16)
    MACRO_LATENT_DIM: int = 16
    NEWS_EMB_DIM: int = 768
    NEWS_PROJ_DIM: int = 64
    FUSION_INPUT_DIM: int = 144  # 64 (Tech) + 16 (Macro) + 64 (News)
    FUSION_NUM_EXPERTS: int = 3  # Dynamic weights: alpha, beta, gamma


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()

