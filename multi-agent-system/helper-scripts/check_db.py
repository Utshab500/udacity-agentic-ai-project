import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("sqlite:///munder_difflin.db")
print(pd.read_sql("SELECT * FROM transactions ORDER BY transaction_date", engine))
# print(pd.read_sql("SELECT * FROM transactions WHERE transaction_date >= '2025-04-01' ORDER BY transaction_date, id", engine))