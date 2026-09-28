import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root: this file is src/msdl_jci/config/settings.py -> parents[3].
BASE_DIR = Path(__file__).resolve().parents[3]
load_dotenv(BASE_DIR / ".env")


# Single source of truth for the SQLite file (env wins, else repo-local default).
_raw_db_path = Path(os.getenv("DATABASE_PATH", "database/main_database.db"))
_SQLITE_DB_PATH = _raw_db_path if _raw_db_path.is_absolute() else BASE_DIR / _raw_db_path


@dataclass(frozen=True)
class Settings:
    """Application settings consumed across the package (paths + ML defaults)."""

    PROJECT_NAME: str = "MSDL-JCI"
    VERSION: str = "0.1.0"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    ROOT_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

    SQLITE_DB_PATH: Path = _SQLITE_DB_PATH

    ML_LOOK_BACK: int = int(os.getenv("ML_LOOK_BACK", "28"))
    ML_PREDICTION_HORIZON: int = int(os.getenv("ML_PREDICTION_HORIZON", "5"))
    ML_HIDDEN_LAYERS: tuple[int, ...] = (50, 25)
    ML_MAX_ITER: int = int(os.getenv("ML_MAX_ITER", "500"))
    ML_RANDOM_STATE: int = int(os.getenv("ML_RANDOM_STATE", "42"))

    NEWS_EMB_DIM: int = 768


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
