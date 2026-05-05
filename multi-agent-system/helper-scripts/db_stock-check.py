import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("sqlite:///munder_difflin.db")

# What does the system think is in stock on April 1, after reorders?
# print(pd.read_sql("""
#     SELECT item_name,
#            SUM(CASE WHEN transaction_type='stock_orders' THEN units ELSE -units END) AS stock
#     FROM transactions
#     WHERE transaction_date <= '2025-04-01'
#       AND item_name IN ('Glossy paper', 'Cardstock', 'Colored paper')
#     GROUP BY item_name
# """, engine))
print(pd.read_sql("""
    SELECT item_name,
           SUM(CASE WHEN transaction_type='stock_orders' THEN units ELSE -units END) AS stock
    FROM transactions
    GROUP BY item_name
""", engine))
# print(pd.read_sql("""
#     SELECT *
#     FROM quotes    
# """, engine))