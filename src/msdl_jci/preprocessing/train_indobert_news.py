import sqlite3

import pandas as pd

MONTH_MAP = {
    'januari': '01', 'jan': '01',
    'februari': '02', 'feb': '02',
    'maret': '03', 'mar': '03',
    'april': '04', 'apr': '04',
    'mei': '05', 'may': '05',
    'juni': '06', 'jun': '06',
    'juli': '07', 'jul': '07',
    'agustus': '08', 'agu': '08', 'aug': '08',
    'september': '09', 'sep': '09',
    'oktober': '10', 'okt': '10', 'oct': '10',
    'november': '11', 'nov': '11',
    'desember': '12', 'des': '12', 'dec': '12'
}


def parse_indonesian_dates(series: pd.Series) -> pd.Series:
    """Vectorized date parser supporting Indonesian and English date formats."""
    extracted = series.astype(str).str.extract(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})')
    day = extracted[0].str.zfill(2)
    month = extracted[1].str.lower().map(MONTH_MAP)
    year = extracted[2]

    date_str = year + '-' + month + '-' + day
    return pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')


def load_and_align_news(db_path: str = "./berita_ihsg_enterprise.db") -> pd.DataFrame:
    query = """
        SELECT title, snippet AS content, published AS raw_date, 'detik' AS source
        FROM detik_ihsg_articles
        UNION ALL
        SELECT title, content, publish_date AS raw_date, 'cnbc' AS source
        FROM cnbc_ihsg_articles
        UNION ALL
        SELECT title, snippet AS content, published AS raw_date, 'kontan' AS source
        FROM kontan_ihsg_articles
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        print("[WARNING] Data berita kosong.")
        return df

    # Parse and clean dates
    df['parsed_date'] = parse_indonesian_dates(df['raw_date'])
    df = df.dropna(subset=['parsed_date']).copy()

    # Temporal Alignment: Vectorized rollover weekend (Saturday/Sunday) to Monday
    weekday = df['parsed_date'].dt.weekday
    days_to_add = pd.to_timedelta((weekday == 5) * 2 + (weekday == 6) * 1, unit='D')
    df['trade_date'] = (df['parsed_date'] + days_to_add).dt.strftime('%Y-%m-%d')

    # Aggregation per trading date
    df_grouped = df.groupby('trade_date', as_index=False).agg({
        'title': lambda x: " | ".join(dict.fromkeys(v.strip() for v in x if pd.notna(v) and str(v).strip())),
        'content': lambda x: " ".join(v.strip() for v in x if pd.notna(v) and str(v).strip()),
        'source': lambda x: ", ".join(sorted(set(str(s) for s in x if pd.notna(s))))
    })

    # Keep clean column ordering
    df_grouped = df_grouped[['trade_date', 'title', 'content', 'source']]

    print(f"[INFO] Berhasil memproses {len(df)} berita dari Kontan, Detik, & CNBC!")
    print(f"[INFO] Total tanggal perdagangan terindeks: {len(df_grouped)}")
    return df_grouped


if __name__ == "__main__":
    processed_news = load_and_align_news("./berita_ihsg_enterprise.db")
    if not processed_news.empty:
        print("\n--- Sample Output (Top 5 Rows) ---")
        print(processed_news[['trade_date', 'source']].head(5))
        output_file = "processed_daily_news.csv"
        processed_news.to_csv(output_file, index=False)
        print(f"\n[SUCCESS] File berhasil disimpan ke '{output_file}'!")
