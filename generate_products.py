import random
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

product_categories = [
    "Electronics",
    "Food & Beverages",
    "Cosmetics",
    "Pharmaceuticals",
    "Home Appliances",
    "Clothing & Apparel",
    "Furniture",
    "Automotive Parts",
    "Books & Stationery",
    "Medical Devices",
    "Toys",
    "Industrial Equipment",
    "Personal Care",
    "Glassware",
    "Fresh Produce"
]

def generate_products(n=100):
    data = []

    for i in range(n):
        row = {
            "product_category": random.choice(product_categories),
            "weight_kg": round(random.uniform(0.1, 20), 2),
            "fragility_level": random.randint(1, 5),
            "shipping_distance_km": round(random.uniform(50, 2000), 2),
        }
        data.append(row)

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_products(100)
    df.to_sql("products", engine, if_exists="replace", index=False)
    print("Inserted 100 product records successfully.")
