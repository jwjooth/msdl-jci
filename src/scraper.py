import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / "development" / ".env"

load_dotenv(dotenv_path=env_path)

db_name = os.getenv("db_name")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")

print(f"Berhasil memuat database: {db_name} dengan user: {db_user} dan password :{db_password}")
