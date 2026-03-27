import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler

print("Script started...")

# -----------------------------
# Database Configuration
# -----------------------------
DB_USER = "postgres"
DB_PASSWORD = "mrbean"   # <-- CHANGE THIS
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecopackai"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    query = "SELECT * FROM materials;"
    df = pd.read_sql(query, engine)
    return df


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_data(df):
    print("Initial Shape:", df.shape)

    # Remove duplicates (ignore auto-generated id & created_at)
    df = df.drop_duplicates(
        subset=[
            "material_name",
            "material_type",
            "strength_rating",
            "weight_capacity_kg",
            "biodegradability_score",
            "co2_emission_score",
            "recyclability_percent",
            "cost_per_unit",
        ]
    )

    print("After Removing Duplicates:", df.shape)

    # Handle missing values (median for numeric)
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print("Missing values after cleaning:")
    print(df.isnull().sum())

    # Outlier Treatment using IQR clipping
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    return df


# -----------------------------
# Feature Engineering
# -----------------------------
def feature_engineering(df):

    # CO2 Impact Index
    df["co2_impact_index"] = df["co2_emission_score"] * (
        1 - df["recyclability_percent"] / 100
    )

    # Cost Efficiency Index
    df["cost_efficiency_index"] = (
        df["strength_rating"] / df["cost_per_unit"]
    )

    # Material Suitability Score
    df["material_suitability_score"] = (
        0.4 * df["strength_rating"]
        + 0.3 * df["biodegradability_score"]
        + 0.3 * df["recyclability_percent"]
    )

    return df


# -----------------------------
# Normalization (Add scaled columns only)
# -----------------------------
def normalize_data(df):

    scaler = StandardScaler()

    numeric_cols = [
        "strength_rating",
        "weight_capacity_kg",
        "biodegradability_score",
        "co2_emission_score",
        "recyclability_percent",
        "cost_per_unit",
    ]

    scaled_values = scaler.fit_transform(df[numeric_cols])

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=[col + "_scaled" for col in numeric_cols]
    )

    df = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    return df


# -----------------------------
# Save Cleaned Data
# -----------------------------
def save_clean_data(df):
    df.to_sql("materials_cleaned", engine, if_exists="replace", index=False)
    print("Cleaned dataset saved to 'materials_cleaned' table.")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df = normalize_data(df)
    save_clean_data(df)

    print("Data Cleaning & Feature Engineering Complete.")
