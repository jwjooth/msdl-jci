"""Core application settings using Pydantic Settings.

Loads configuration from environment variables and .env files.
This is the single source of truth for all app-level configuration.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Resolve .env relative to project root (<project>/development/.env or root/.env)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

for _env_candidate in (
    _PROJECT_ROOT / ".env",
    _PROJECT_ROOT / "development" / ".env",
):
    if _env_candidate.exists():
        load_dotenv(dotenv_path=_env_candidate, override=False)
        break


class Settings(BaseSettings):
    """Application settings.

    Environment variables override these defaults. The `.env` file is also
    consulted at import time.
    """

    # ── Project ──────────────────────────────────────────────
    PROJECT_NAME: str = "MSDL-JCI"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── Paths ────────────────────────────────────────────────
    DATA_RAW_DIR: Path = _PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = _PROJECT_ROOT / "data" / "processed"

    # ── Database (PostgreSQL) ────────────────────────────────
    DB_NAME: str = os.getenv("db_name", "msdl-jci")
    DB_USER: str = os.getenv("db_user", "postgres")
    DB_PASSWORD: str = os.getenv("db_password", "postgres")
    DB_HOST: str = os.getenv("db_host", "localhost")
    DB_PORT: str = os.getenv("db_port", "5432")

    DB_POOL_MIN: int = 2
    DB_POOL_MAX: int = 10

    @property
    def DATABASE_URL(self) -> str:
        """Return a PostgreSQL connection URL."""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        """Return a psycopg2 / psycopg compatible connection string."""
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ── ML ───────────────────────────────────────────────────
    ML_LOOK_BACK: int = 3
    ML_HIDDEN_LAYERS: tuple[int, ...] = (50, 25)
    ML_MAX_ITER: int = 500
    ML_RANDOM_STATE: int = 42

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache avoids re-parsing the environment on every call.
    """
    return Settings()
