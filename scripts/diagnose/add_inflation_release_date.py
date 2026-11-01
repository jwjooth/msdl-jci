"""Add release_date column to inflation_data with actual BPS press release dates.

BPS releases inflation data 1st-5th of the FOLLOWING month.
E.g., "Desember 2025" data released ~2026-01-02 to 2026-01-05.
We approximate with the 2nd business day of the following month.
"""
import re
import sqlite3
import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import BDay

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "database" / "main_database.db"

_ID_MONTH = {
    "januari": "January", "februari": "February", "maret": "March",
    "april": "April", "mei": "May", "juni": "June", "juli": "July",
    "agustus": "August", "september": "September", "oktober": "October",
    "november": "November", "desember": "December",
}


def period_to_release_date(periode: str) -> pd.Timestamp:
    """Convert Indonesian period string to approximate BPS release date.
    
    BPS releases inflation 1st-5th of following month.
    We use 2nd business day of following month as approximation.
    """
    s = str(periode).strip().lower()
    for id_month, en_month in _ID_MONTH.items():
        s = re.sub(rf"\b{id_month}\b", en_month, s)
    dt = pd.to_datetime(s, format="%B %Y")
    first_next_month = (dt + pd.DateOffset(months=1)).replace(day=1)
    return first_next_month + BDay(1)


def main():
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()
    
    cursor.execute("PRAGMA table_info(inflation_data)")
    cols = [c[1] for c in cursor.fetchall()]
    if "release_date" not in cols:
        cursor.execute("ALTER TABLE inflation_data ADD COLUMN release_date TEXT")
        con.commit()
        print("Added release_date column")
    else:
        print("release_date column already exists")
    
    df = pd.read_sql_query("SELECT * FROM inflation_data", con)
    missing = df[df["release_date"].isna()]
    
    dates = missing["Periode"].apply(period_to_release_date).dt.strftime("%Y-%m-%d")
    
    for periode, release_date in zip(missing["Periode"], dates):
        cursor.execute(
            "UPDATE inflation_data SET release_date = ? WHERE Periode = ?",
            (release_date, periode)
        )
    
    con.commit()
    con.close()
    print("Done. release_date populated.")


if __name__ == "__main__":
    main()
