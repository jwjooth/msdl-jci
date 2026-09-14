# Data schemas

## data/raw/technical/jci_historical.csv
| column | type | format |
|---|---|---|
| Date | date | YYYY-MM-DD |
| Open, High, Low, Close | float | index points |
| Volume | float | shares |

## data/raw/macro/bi_rate.csv
Period (Indonesian month text, e.g. "Januari 2020"), `BI-7Day-RR` ("6,00%" style).
Parsed by `parse_indonesian_date`. **Point-in-time:** period dates, not release
dates — treat as available the 1st of the next month (conservative).

## data/raw/macro/inflation_data.csv
`Periode`, `Data Inflasi` — same caveat as BI rate.

## data/raw/macro/kurs_usdidr.csv
`Date` (YYYY-MM-DD), `Close` (may contain commas).

## data/processed/daily_news_embeddings.csv
`trade_date` (YYYY-MM-DD) + `emb_0..emb_767` (Frozen IndoBERT, 768 dims).
Missing dates → zero-vector imputation at merge; exact-date left join (no leakage).

## data/sample/aligned_sample.csv
First 60 aligned rows (8 embedding dims) for CI smoke tests. Full data is git-ignored.
