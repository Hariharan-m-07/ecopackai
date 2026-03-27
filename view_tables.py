import pandas as pd
import os
from sqlalchemy import create_engine

print("Script started...")
print("Current Working Directory:", os.getcwd())

# -----------------------------
# Database Configuration
# -----------------------------
DB_USER = "postgres"
DB_PASSWORD = "mrbean"  
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecopackai"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# -----------------------------
# Load Tables
# -----------------------------
raw_df = pd.read_sql("SELECT * FROM materials LIMIT 50;", engine)
clean_df = pd.read_sql("SELECT * FROM materials_cleaned LIMIT 50;", engine)

print("\n===== RAW TABLE (FIRST 5 ROWS) =====\n")
print(raw_df.head())

print("\n===== CLEANED TABLE (FIRST 5 ROWS) =====\n")
print(clean_df.head())

# -----------------------------
# Export CSV
# -----------------------------
raw_df.to_csv("raw_materials_sample.csv", index=False)
clean_df.to_csv("clean_materials_sample.csv", index=False)

print("\nCSV files exported successfully.")
