from datetime import datetime
import logging
from typing import Dict, List, Optional
import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extensions import connection as PGConnection

# Konfigurasi Logging standar industri
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NewsDatabaseManager:
  """Kelas pengelola koneksi dan operasi database PostgreSQL."""

  def __init__(self, db_config: Dict[str, str]):
    self.db_config = db_config

  def _get_connection(self) -> PGConnection:
    """Membuat koneksi ke database PostgreSQL."""
    return psycopg2.connect(**self.db_config)

  def save_articles(self, articles: List[Dict[str, any]]) -> None:
    """Menyimpan daftar artikel berita ke PostgreSQL secara massal

    dengan mencegah duplikasi URL (UPSERT/DO NOTHING).
    """
    insert_query = """
            INSERT INTO financial_news (url, source, title, content, published_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING;
        """

    # Menggunakan Context Manager agar koneksi otomatis tertutup (aman dari memory leak)
    try:
      with self._get_connection() as conn:
        with conn.cursor() as cursor:
          # Konversi list dictionary ke tuple untuk eksekusi batch
          data_tuples = [
              (
                  item["url"],
                  item["source"],
                  item["title"],
                  item["content"],
                  item["published_date"],
              )
              for item in articles
          ]

          psycopg2.extras.execute_batch(cursor, insert_query, data_tuples)
          conn.commit()
          logger.info(
              f"Berhasil memproses {len(articles)} artikel ke database."
          )

    except psycopg2.Error as e:
      logger.error(f"Gagal menyimpan data ke database: {e}")
      raise


# --- Contoh Mock Scraper / Fungsi Pengumpul Berita ---
def fetch_news_for_date(target_date: datetime) -> List[Dict[str, any]]:
  """Fungsi modular untuk mengambil berita pada tanggal tertentu.

  (Ganti bagian ini dengan logika BeautifulSoup/Selenium/API Anda).
  """
  # Simulasi hasil scraping harian dari 3 sumber
  mock_articles = [
      {
          "url": f"https://finance.detik.com/bursa/d-12345/{target_date.strftime('%Y%m%d')}",
          "source": "detik",
          "title": "IHSG Ditutup Menguat Hari Ini",
          "content": (
              "Indeks Harga Saham Gabungan (IHSG) bergerak di zona hijau..."
          ),
          "published_date": target_date,
      }
  ]
  return mock_articles


# --- Main Pipeline Execution ---
if __name__ == "__main__":
  # Konfigurasi Database (Gunakan Environment Variables pada production)
  DB_CONFIG = {
      "dbname": "db_jci_prediction",
      "user": "postgres",
      "password": "your_secure_password",
      "host": "localhost",
      "port": "5432",
  }

  db_manager = NewsDatabaseManager(DB_CONFIG)

  # Simulasi rentang tanggal (Contoh: 1 hari spesifik)
  sample_date = datetime(2026, 6, 8, 16, 0, 0)

  logger.info("Memulai proses pengumpulan berita...")
  daily_news = fetch_news_for_date(sample_date)

  if daily_news:
    db_manager.save_articles(daily_news)
  else:
    logger.warning(
        f"Tidak ada berita ditemukan untuk tanggal {sample_date.date()}"
    )