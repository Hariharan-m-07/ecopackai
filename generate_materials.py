import random
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

print("Material generation started...")

# -----------------------------
# Database Configuration
# -----------------------------
DB_USER = "postgres"
DB_PASSWORD = "mrbean"   # <-- PUT YOUR REAL PASSWORD
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecopackai"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# -----------------------------
# Material Categories (imbalanced distribution)
# -----------------------------
material_types = (
    ["Recycled Paper"] * 200 +
    ["Bioplastic"] * 150 +
    ["Molded Fiber"] * 120 +
    ["Compostable Polymer"] * 100 +
    ["Glass"] * 80 +
    ["Aluminum"] * 70 +
    ["Bamboo"] * 50 +
    ["Recycled Plastic"] * 30
)

# -----------------------------
# Generate Realistic Dataset
# -----------------------------
def generate_material_data(n=850):
    data = []

    for i in range(n):
        material_type = random.choice(material_types)

        # Base features
        strength = round(random.uniform(4.0, 10.0), 2)
        weight = round(random.uniform(3, 60), 2)
        biodeg = round(random.uniform(3.0, 10.0), 2)
        recycle = round(random.uniform(30, 100), 2)

        # COST depends on strength + weight
        cost = round(
            0.6 * strength +
            0.04 * weight +
            random.uniform(0.5, 1.5),
            2
        )

        # CO2 depends inversely on recyclability & biodegradability
        co2 = round(
            12 -
            0.06 * recycle -
            0.4 * biodeg +
            random.uniform(0, 1.2),
            2
        )

        row = {
            "material_name": f"{material_type} Material {i+1}",
            "material_type": material_type,
            "strength_rating": strength,
            "weight_capacity_kg": weight,
            "biodegradability_score": biodeg,
            "co2_emission_score": max(co2, 0.5),
            "recyclability_percent": recycle,
            "cost_per_unit": max(cost, 0.5)
        }

        data.append(row)

    df = pd.DataFrame(data)

    # Introduce Missing Values (7%)
    for col in df.columns:
        if col not in ["material_name", "material_type"]:
            df.loc[df.sample(frac=0.07, random_state=42).index, col] = np.nan

    # Introduce Outliers (2%)
    outlier_indices = df.sample(frac=0.02, random_state=42).index
    df.loc[outlier_indices, "cost_per_unit"] *= 3
    df.loc[outlier_indices, "co2_emission_score"] *= 2

    # Introduce Duplicates (5%)
    duplicate_rows = df.sample(frac=0.05, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


# -----------------------------
# Insert into Database
# -----------------------------
def insert_into_db(df):
    df.to_sql("materials", engine, if_exists="replace", index=False)
    print(f"Inserted {len(df)} materials successfully.")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    df_materials = generate_material_data()
    insert_into_db(df_materials)
    print("Material generation completed.")