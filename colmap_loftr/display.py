import sqlite3
import pandas as pd

con = sqlite3.connect('database.db')
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
tables = [table[0] for table in tables]

print(f'Tables: {tables}')
for table in tables:
    data = pd.read_sql_query(f'SELECT * FROM {table}', con)
    print(f'{table}:')
    print(f'{data}')