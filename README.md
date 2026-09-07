# MSDL-JCI (Multi-Source Deep Learning for Jakarta Composite Index)

Kerangka kerja modular Python berstandar industri (*src-layout*) untuk ekstraksi representasi multi-sumber, feature engineering data makroekonomi, teknikal pasar modal, dan sentimen/embedding berita finansial guna analisis IHSG (Jakarta Composite Index).

---

## 🏛️ Arsitektur Direktori Standar Industri

Struktur proyek dirancang bersih dan modular mengikuti best practice Python packaging modern (PEP 518, PEP 621, and `src-layout`):

```text
MSDL-JCI/
├── .env.example                # Template konfigurasi environment variable
├── .gitignore                  # Aturan ignore git standar (venv, pycache, tests, db)
├── pyproject.toml              # Definisi project metadata, dependencies, dan entrypoint
├── README.md                   # Dokumentasi arsitektur dan panduan operasional
├── uv.lock                     # Lockfile resolusi dependency dari uv
│
├── data/                       # Data storage (terpisah dari logic kode)
│   ├── database/               # Database SQLite / Relational lokal
│   │   └── berita_ihsg_enterprise.db
│   ├── processed/              # Data hasil transformasi / feature embeddings
│   │   ├── daily_news_embeddings.csv
│   │   └── processed_daily_news.csv
│   └── raw/                    # Raw historical dataset
│       ├── macro/              # Data makroekonomi (BI-rate, inflasi, kurs USD/IDR)
│       │   ├── bi_rate.csv
│       │   ├── inflation_data.csv
│       │   └── kurs_usdidr.csv
│       └── technical/          # Data teknikal IHSG (OHLCV)
│           └── jci_historical.csv
│
├── src/
│   └── msdl_jci/               # Package utama MSDL-JCI
│       ├── __init__.py         # Package root exports
│       ├── main.py             # CLI Entrypoint aplikasi
│       │
│       ├── config/             # Manajemen Konfigurasi Terpusat
│       │   ├── __init__.py
│       │   └── settings.py     # Settings class, dynamic path resolution, env loader
│       │
│       ├── utils/              # Helper Umum & Reusable Modules
│       │   ├── __init__.py
│       │   ├── data_loader.py  # Loader CSV & numeric series parser yang aman
│       │   └── logging_config.py # Standardized stream logger
│       │
│       ├── models/             # Modul Ekstraksi Representasi & Pipeline
│       │   ├── __init__.py
│       │   ├── macro/          # Macroeconomic MLP Encoder & Pipeline
│       │   │   ├── __init__.py
│       │   │   ├── encoder.py
│       │   │   └── pipeline.py
│       │   ├── technical/      # Technical Price MLP Encoder & Pipeline
│       │   │   ├── __init__.py
│       │   │   ├── encoder.py
│       │   │   └── pipeline.py
│       │   └── news/           # Financial News Representation Model & Pipeline
│       │       ├── __init__.py
│       │       ├── encoder.py
│       │       └── pipeline.py
│       │
│       ├── preprocessing/      # Data cleansing & alignment pipeline
│       │   ├── __init__.py
│       │   ├── extract_embeddings.py
│       │   ├── train_indobert_news.py
│       │   ├── train_lstm_jci.py
│       │   └── train_mlp_macro.py
│       │
│       └── scraping/           # News scraper crawlers
│           ├── __init__.py
│           ├── scraping_cnbc_indonesia_news.py
│           ├── scraping_detik_news.py
│           └── scraping_kontan_news.py
│
└── tests/                      # Automated Testing Suite (Pytest)
    ├── __init__.py
    ├── integration/            # Pengujian integrasi end-to-end
    │   ├── __init__.py
    │   └── test_pipeline.py
    └── unit/                   # Pengujian unit per komponen
        ├── __init__.py
        ├── test_config.py
        ├── test_data_loader.py
        ├── test_macro_encoder.py
        └── test_technical_encoder.py
```

---

## ⚙️ Konfigurasi Sistem Terpusat (`src/msdl_jci/config/settings.py`)

Aplikasi menggunakan `Settings` berbasis dataclass dengan caching (`lru_cache`) yang otomatis:
- Mendeteksi `PROJECT_ROOT` tanpa hardcoding relative path manual.
- Mengarahkan path dataset default ke folder `data/raw/macro/`, `data/raw/technical/`, dan `data/processed/`.
- Membaca konfigurasi dari file `.env` jika tersedia.

Untuk kustomisasi konfigurasi:
1. Salin `.env.example` ke `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Sesuaikan variabel seperti `ML_LOOK_BACK`, `LOG_LEVEL`, atau kredensial database jika menggunakan PostgreSQL.

---

## 🚀 Panduan Instalasi & Eksekusi

### 1. Prasyarat
- Python `>= 3.12`
- Paket manajer modern: [`uv`](https://github.com/astral-sh/uv)

### 2. Instalasi Dependencies
```powershell
uv sync
```

### 3. Menjalankan Pipeline Utama

Anda dapat menjalankan pipeline melalui command line script atau Python module runner:

**Opsi A (CLI Console Script):**
```powershell
uv run msdl-jci
```

**Opsi B (Python Module):**
```powershell
uv run python -m msdl_jci.main
```

---

## 🧪 Pengujian Otomatis (Testing)

Proyek ini telah dilengkapi dengan unit test dan integration test untuk menjamin keandalan fungsionalitas modul dan koneksi pipeline data.

Jalankan test suite dengan:
```powershell
uv run pytest -v
```

Untuk melihat cakupan kode (coverage report):
```powershell
uv run pytest --cov=msdl_jci
```

---

## 🛠️ Modul & Komponen Kunci

- **`msdl_jci.config.settings`**: Sumber kebenaran tunggal (*single source of truth*) untuk direktori proyek, path file dataset, dan parameter machine learning.
- **`msdl_jci.utils.data_loader`**: Fungsi utilitas untuk membaca, memvalidasi kolom, dan membersihkan format numerik dataset time-series.
- **`msdl_jci.models.macro`**: Menghandle pelatihan regresi non-linear dan pembentukan embedding gabungan dari BI-7Day-RR, inflasi, dan kurs USD/IDR.
- **`msdl_jci.models.technical`**: Mengonstruksi regresi sekuensial pada harga penutupan IHSG historis.
- **`msdl_jci.models.news`**: Menyediakan arsitektur pipeline untuk pemodelan embedding berita finansial hasil ekstraksi IndoBERT.
