"""News corpus audit (READ-ONLY): per-portal coverage, years, IHSG relevance,
sectoral keyword density. Uses the notebook's own clean_day logic.
Run: python scripts/diagnose/news_corpus_audit.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from msdl_dates import clean_day  # single copy shared with notebooks/development.ipynb


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
