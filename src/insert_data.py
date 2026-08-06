import pandas as pd
from sqlalchemy import create_engine

# read csv file
file_path = (
    # r"C:\Users\jorda\Documents\'kuliah hohohoho'\'semester 7'\project\MSDL-JCI\src\jci_historical.csv"
    r"C:\Users\jorda\Documents\kuliah hohohoho\semester 7\project\MSDL-JCI\data\raw\bi_rate.csv"
)
df = pd.read_csv(file_path)

# connection database
engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/msdl-jci"
)

# put it into database table
df.to_sql(
    "interest_rate",
    con=engine,
    schema="macro_data",
    if_exists="append",
    index=False,
)

print("Data CSV berhasil dimasukkan via Python!")
