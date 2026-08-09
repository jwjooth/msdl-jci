"""Application settings."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    DEBUG: bool = False

    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

    DB_NAME: str = os.getenv("db_name", "msdl-jci")
    DB_USER: str = os.getenv("db_user", "postgres")
    DB_PASSWORD: str = os.getenv("db_password", "postgres")
    DB_HOST: str = os.getenv("db_host", "localhost")
    DB_PORT: str = os.getenv("db_port", "5432")

    ML_LOOK_BACK: int = 3
    ML_HIDDEN_LAYERS: tuple[int, ...] = (50, 25)
    ML_MAX_ITER: int = 500
    ML_RANDOM_STATE: int = 42

    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
