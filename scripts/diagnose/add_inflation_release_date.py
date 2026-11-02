"""Add release_date to inflation_data and publication_date to bi_rate.

Issue #61: Inflation uses period label (first-of-month) instead of actual BPS release date
(1st-5th of following month). BPS "Desember 2025" data released ~2026-01-02 to 2026-01-05.
We approximate with 2nd business day of following month.

Issue #59: BI rate uses meeting date instead of actual publication date.
BI announces rate decisions same day as the meeting — but notebook ENTITY_TABLES expects publication_date.

Both fixes align macro data with actual release dates, eliminating forward-leak bias.
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "database" / "main_database.db"

_ID_MONTH = {
    "januari": "January", "februari": "February", "maret": "March",
    "april": "April", "mei": "May", "juni": "June", "juli": "July",
    "agustus": "August", "september": "September", "oktober": "October",
    "november": "November", "descemer": "December",
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


def bi_date_to_publication_date(period_str: str) -> pd.Timestamp:
    """BI meeting date string (e.g., "17 Desember 2025") -> same-day publication date.

    BI announces rate decisions same day as the meeting, so publication_date == meeting date.
    """
    s = str(period_str).strip().lower()
    for id_month, en_month in _ID_MONTH.items():
        s = re.sub(rf"\b{id_month}\b", en_month, s)
    return pd.to_datetime(s, format="%d %B %Y")


def main():
    import shutil
    import sys

    if "--write" not in sys.argv:
        print("DRY RUN (read-only). Re-run with --write to mutate the DB.")
        print(f"Would add release_date to inflation_data and publication_date to bi_rate in {DB_PATH}")
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        for tbl, col in (("inflation_data", "release_date"), ("bi_rate", "publication_date")):
            cur.execute(f"PRAGMA table_info({tbl})")
            cols = [c[1] for c in cur.fetchall()]
            print(f"  {tbl}.{col}: {'exists' if col in cols else 'MISSING (would add)'}")
        con.close()
        return

    backup = DB_PATH.with_suffix(DB_PATH.suffix + ".bak")
    shutil.copy2(DB_PATH, backup)
    print(f"backup written: {backup}")

    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()

    # --- inflation_data: release_date ---
    cursor.execute("PRAGMA table_info(inflation_data)")
    cols = [c[1] for c in cursor.fetchall()]
    if "release_date" not in cols:
        cursor.execute("ALTER TABLE inflation_data ADD COLUMN release_date TEXT")
        con.commit()
        print("Added release_date column to inflation_data")
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

    print(f"Populated release_date for {len(missing)} inflation rows")

    # --- bi_rate: publication_date ---
    cursor.execute("PRAGMA table_info(bi_rate)")
    cols = [c[1] for c in cursor.fetchall()]
    if "publication_date" not in cols:
        cursor.execute("ALTER TABLE bi_rate ADD COLUMN publication_date TEXT")
        con.commit()
        print("Added publication_date column to bi_rate")
    else:
        print("publication_date column already exists")

    df_bi = pd.read_sql_query("SELECT * FROM bi_rate", con)
    missing_bi = df_bi[df_bi["publication_date"].isna()]

    pub_dates = missing_bi["Period"].apply(bi_date_to_publication_date).dt.strftime("%Y-%m-%d")

    for periode, pub_date in zip(missing_bi["Period"], pub_dates):
        cursor.execute(
            "UPDATE bi_rate SET publication_date = ? WHERE Period = ?",
            (pub_date, periode)
        )

    print(f"Populated publication_date for {len(missing_bi)} bi_rate rows")

    con.commit()
    con.close()
    print("Done.")


if __name__ == "__main__":
    main()