"""Regenerate thesis Tables 1-4 straight from database/main_database.db.

READ-ONLY. Run: python scripts/diagnose/gen_sample_tables.py
Output: markdown-ready sample rows + range checks.
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "database" / "main_database.db")

j = pd.read_sql_query("SELECT * FROM jci_historical", con)
print(f"JCI rows={len(j)} range={j['Date'].min()} -> {j['Date'].max()}")
print("TABLE 1 (first 5 trading days):")
print(j.head(5).to_string(index=False))

k = pd.read_sql_query("SELECT Date, Close FROM kurs_usdidr", con)
k["Date"] = pd.to_datetime(k["Date"], format="mixed", errors="coerce")
k = k.sort_values("Date").reset_index(drop=True)
print(f"\nKURS rows={len(k)} range={k['Date'].min().date()} -> {k['Date'].max().date()}")
print("TABLE 2 (last 5 rows, latest data):")
print(k.tail(5).to_string(index=False))

b = pd.read_sql_query("SELECT * FROM bi_rate", con)
print(f"\nBI rows={len(b)}")
print("TABLE 3 (latest 5 releases):")
print(b.head(5).to_string(index=False))

f = pd.read_sql_query("SELECT * FROM inflation_data", con)
print(f"\nINFLATION rows={len(f)}")
print("TABLE 4 (Jan-May 2025, as in thesis):")
print(f.iloc[11:6:-1].to_string(index=False))
print("\nDONE")
