"""News corpus audit (READ-ONLY): per-portal coverage, years, IHSG relevance,
sectoral keyword density. Uses the notebook's own clean_day logic.
Run: python scripts/diagnose/news_corpus_audit.py
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "database" / "main_database.db")

_ID_MONTH = {"januari": "01", "februari": "02", "maret": "03", "april": "04",
             "mei": "05", "juni": "06", "juli": "07", "agustus": "08",
             "september": "09", "oktober": "10", "november": "11", "desember": "12",
             "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
             "jul": "07", "agu": "08", "ags": "08", "sep": "09", "okt": "10",
             "nov": "11", "des": "12"}


def clean_day(s):  # mirror of load_news.clean_day in notebooks/development.ipynb
    s = str(s).lower().replace(" wib", "").replace("|", " ").replace("/", " ")
    s = re.sub(r"\s*[+-]\d{2}:?\d{2}\s*$", "", s)
    s = re.sub(r"(?<=\d)t(?=\d)", " ", s)
    s = re.sub(r"^(senin|selasa|rabu|kamis|jumat|sabtu|minggu),?\s*", "", s).strip()
    for _k, _v in _ID_MONTH.items():
        s = re.sub(rf"\b{_k}\b", _v, s)
    return s


SECTORS = {
    "bank/suku-bunga": ["bank ", "bca", "bri", "mandiri", "bni", "suku bunga"],
    "asing/flow": ["asing", "net buy", "net sell", "arus modal", "capital outflow"],
    "moneter/bi": ["bank indonesia", "bi-rate", "rdg"],
    "lq45/emiten": ["lq45", "emiten", "dividen", "laba bersih"],
}

for t, dcol, tcol, bcol, ccol in [
    ("detik_ihsg_articles", "published", "title", "snippet", "category"),
    ("kontan_ihsg_articles", "published", "title", "snippet", "category"),
    ("cnbc_ihsg_articles", "publish_date", "title", "content", None),
]:
    d = pd.read_sql_query(f"SELECT * FROM [{t}]", con)
    dt = pd.to_datetime(d[dcol].astype(str).map(clean_day),
                        format="mixed", errors="coerce", dayfirst=True)
    txt = (d[tcol].fillna("") + " " + d[bcol].fillna("")).str.lower()
    empty = txt.str.strip().eq("").sum()
    print(f"--- {t}: n={len(d)} range={dt.min().date()} -> {dt.max().date()} "
          f"unparseable={int(dt.isna().sum())} empty_text={int(empty)}")
    print(f"    ihsg_mention={txt.str.contains('ihsg').mean():.3f}")
    print(f"    per_year={dt.dt.year.value_counts().sort_index().to_dict()}")
    if ccol:
        print(f"    categories={d[ccol].value_counts().head(6).to_dict()}")
    print("    sectors=" + str(
        {k: round(float(txt.str.contains('|'.join(v)).mean()), 3)
         for k, v in SECTORS.items()}))
print("DONE")
