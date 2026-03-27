import pandas as pd
from sqlalchemy import create_engine

DB_USER = "postgres"
DB_PASSWORD = "mrbean"   # <-- CHANGE THIS
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecopackai"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

df = pd.read_sql("SELECT * FROM products;", engine)

print("\n===== PRODUCTS TABLE (FIRST 20 ROWS) =====\n")
print(df)
df.to_csv("products_sample.csv", index=False)
