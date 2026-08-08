import yfinance as yf

# pull IHSG data from yfinance library
# jci_df = yf.download("^JKSE", start="2018-01-01", end="2025-12-31")
usdidr_df = yf.download("USDIDR=X", start="2018-01-01", end="2025-12-31")

# convert to csv before import to database
# jci_df.to_csv("jci_historical.csv")
usdidr_df.to_csv("kurs_usdidr.csv")
