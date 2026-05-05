import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("sqlite:///munder_difflin.db")

# What does the system think is in stock on April 1, after reorders?
print(pd.read_sql("""
       SELECT transaction_date, item_name, transaction_type, units, price
       FROM transactions
       WHERE transaction_date >= '2025-04-01'
       ORDER BY transaction_date, id
   """, engine))