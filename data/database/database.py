import os
from dotenv import load_dotenv
from pathlib import Path
from psycopg_pool import ConnectionPool

env_path = Path(__file__).resolve().parent / "development" / ".env"
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv("db_name")
DB_USER = os.getenv("db_user")
DB_PASSWORD = os.getenv("db_password")
DB_HOST = os.getenv("db_host", "localhost")
DB_PORT = os.getenv("db_port", "5432")

# Membuat string koneksi (DSN Connection String)
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Inisialisasi Connection Pool (Standar Industri untuk Performa Gahar)
# min_size: Jumlah koneksi minimal yang dijaga tetap hidup (warm connections)
# max_size: Batas maksimal koneksi saat load tinggi
db_pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=2,
    max_size=10,
    open=True
)

def get_db_connection():
    """
    Mengambil koneksi instan dari pool agar proses gesit tanpa overhead.
    Gunakan konteks 'with' agar koneksi otomatis dikembalikan ke pool setelah selesai.
    """
    return db_pool.connection()

# Contoh cara pakai di skrip utama Anda:
if __name__ == "__main__":
    try:
        # Membuka koneksi instan dari pool
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                db_version = cursor.fetchone()
                print(f"Koneksi Berhasil! PostgreSQL Version: {db_version[0]}")
    except Exception as e:
        print(f"Koneksi Gagal: {e}")
    finally:
        # Menutup pool secara bersih saat aplikasi dimatikan
        db_pool.close()