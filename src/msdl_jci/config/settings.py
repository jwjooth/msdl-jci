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

    # Database Configuration (PostgreSQL / Relational)
    DB_NAME: str = os.getenv("DB_NAME", os.getenv("db_name", "msdl-jci"))
    DB_USER: str = os.getenv("DB_USER", os.getenv("db_user", "postgres"))
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", os.getenv("db_password", "postgres"))
    DB_HOST: str = os.getenv("DB_HOST", os.getenv("db_host", "localhost"))
    DB_PORT: str = os.getenv("DB_PORT", os.getenv("db_port", "5432"))

    # Machine Learning / Modeling Hyperparameters
    ML_LOOK_BACK: int = int(os.getenv("ML_LOOK_BACK", "3"))
    ML_HIDDEN_LAYERS: tuple[int, ...] = (50, 25)
    ML_MAX_ITER: int = int(os.getenv("ML_MAX_ITER", "500"))
    ML_RANDOM_STATE: int = int(os.getenv("ML_RANDOM_STATE", "42"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
