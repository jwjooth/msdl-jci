import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    _raw_db_path = os.getenv("DATABASE_PATH", "./database/main_database.db")

    if Path(_raw_db_path).is_absolute():
        DB_FULL_PATH = Path(_raw_db_path)
    else:
        DB_FULL_PATH = BASE_DIR / _raw_db_path

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def get_db_url(cls) -> str:
        return f"sqlite:///{cls.DB_FULL_PATH}"

    @classmethod
    def validate_db_exists(cls):
        if not cls.DB_FULL_PATH.exists():
            raise FileNotFoundError(
                f"Database file not found at {cls.DB_FULL_PATH}\n"
                f"Check your '.env' file or ensure the './database/' folder exists."
            )
