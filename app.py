import io
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

app = Flask(__name__)

# -----------------------------
# DATABASE CONFIG
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Local fallback
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "mrbean")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "ecopackai")
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
else:
    # Render/hosted fallback normalization
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine: Engine = create_engine(DATABASE_URL)

# -----------------------------
# LOAD MODELS
# -----------------------------
cost_model = joblib.load("cost_model.pkl")
co2_model = joblib.load("co2_model.pkl")


# -----------------------------
# DATA BOOTSTRAP
# -----------------------------
MATERIAL_TYPES = [
    "Bioplastic",
    "Recycled Paper",
    "Molded Fiber",
    "Compostable Polymer",
    "Glass",
    "Aluminum",
    "Bamboo",
    "Recycled Plastic",
]


def table_exists(table_name: str) -> bool:
    inspector = inspect(engine)
    return inspector.has_table(table_name)


def generate_seed_data(n: int = 850) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    material_choices = rng.choice(
        MATERIAL_TYPES,
        size=n,
        p=np.array([0.18, 0.24, 0.14, 0.12, 0.08, 0.08, 0.08, 0.08]),
    )

    strengths = rng.uniform(4.0, 10.0, size=n).round(2)
    weights = rng.uniform(3.0, 60.0, size=n).round(2)
    biodeg = rng.uniform(3.0, 10.0, size=n).round(2)
    recycle = rng.uniform(30.0, 100.0, size=n).round(2)

    cost = (0.6 * strengths + 0.04 * weights + rng.uniform(0.5, 1.5, size=n)).round(2)
    co2 = (
        12 - 0.06 * recycle - 0.4 * biodeg + rng.uniform(0.0, 1.2, size=n)
    ).round(2)
    co2 = np.maximum(co2, 0.5)
    cost = np.maximum(cost, 0.5)

    df = pd.DataFrame(
        {
            "material_name": [f"{m} Material {i+1}" for i, m in enumerate(material_choices)],
            "material_type": material_choices,
            "strength_rating": strengths,
            "weight_capacity_kg": weights,
            "biodegradability_score": biodeg,
            "co2_emission_score": co2,
            "recyclability_percent": recycle,
            "cost_per_unit": cost,
        }
    )

    # Add a few duplicates to resemble your project setup
    dup = df.sample(frac=0.05, random_state=42)
    df = pd.concat([df, dup], ignore_index=True)

    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
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
    ).copy()

    numeric_cols = [
        "strength_rating",
        "weight_capacity_kg",
        "biodegradability_score",
        "co2_emission_score",
        "recyclability_percent",
        "cost_per_unit",
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Engineered features
    df["co2_impact_index"] = df["co2_emission_score"] * (
        1 - df["recyclability_percent"] / 100.0
    )
    df["cost_efficiency_index"] = df["strength_rating"] / df["cost_per_unit"]
    df["material_suitability_score"] = (
        0.4 * df["strength_rating"]
        + 0.3 * df["biodegradability_score"]
        + 0.3 * df["recyclability_percent"]
    )

    # Add scaled copies, keep originals
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0) or 1.0
        df[f"{col}_scaled"] = (df[col] - mean) / std

    return df.reset_index(drop=True)


def ensure_data() -> None:
    if table_exists("materials_cleaned"):
        return

    raw_df = generate_seed_data()
    clean_df = clean_and_engineer(raw_df)

    raw_df.to_sql("materials", engine, if_exists="replace", index=False)
    clean_df.to_sql("materials_cleaned", engine, if_exists="replace", index=False)


ensure_data()


# -----------------------------
# HELPERS
# -----------------------------
def build_input_df(
    material_type: str,
    strength: float,
    weight: float,
    biodeg: float,
    recycle: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "material_type": [material_type],
            "strength_rating": [strength],
            "weight_capacity_kg": [weight],
            "biodegradability_score": [biodeg],
            "recyclability_percent": [recycle],
            "co2_impact_index": [0.0],  # placeholder, model ignores if not trained on it
            "cost_efficiency_index": [0.0],
            "material_suitability_score": [0.0],
        }
    )


def get_recommendations_df() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM materials_cleaned", engine)

    model_features = [
        "material_type",
        "strength_rating",
        "weight_capacity_kg",
        "biodegradability_score",
        "recyclability_percent",
        "co2_impact_index",
        "cost_efficiency_index",
        "material_suitability_score",
    ]

    df["predicted_cost"] = cost_model.predict(df[model_features])
    df["predicted_co2"] = co2_model.predict(df[model_features])
    df["eco_score"] = (1 / df["predicted_cost"] + 1 / df["predicted_co2"]) / 2

    return df


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)

    material_type = data.get("material_type", "Bioplastic")
    strength = float(data["strength"])
    weight = float(data["weight"])
    biodeg = float(data["biodegradability"])
    recycle = float(data["recyclability"])

    input_data = build_input_df(material_type, strength, weight, biodeg, recycle)

    model_features = [
        "material_type",
        "strength_rating",
        "weight_capacity_kg",
        "biodegradability_score",
        "recyclability_percent",
        "co2_impact_index",
        "cost_efficiency_index",
        "material_suitability_score",
    ]

    # Recompute engineered features for user input
    input_data["co2_impact_index"] = input_data["biodegradability_score"] * (
        1 - input_data["recyclability_percent"] / 100.0
    )
    input_data["cost_efficiency_index"] = (
        input_data["strength_rating"] / max(float(weight), 1.0)
    )
    input_data["material_suitability_score"] = (
        0.4 * input_data["strength_rating"]
        + 0.3 * input_data["biodegradability_score"]
        + 0.3 * input_data["recyclability_percent"]
    )

    predicted_cost = float(cost_model.predict(input_data[model_features])[0])
    predicted_co2 = float(co2_model.predict(input_data[model_features])[0])
    eco_score = float((1 / predicted_cost + 1 / predicted_co2) / 2)

    df = get_recommendations_df()
    top5 = df.sort_values(by="eco_score", ascending=False).head(5)

    recommendations = []
    for _, row in top5.iterrows():
        recommendations.append(
            {
                "material_type": str(row["material_type"]),
                "predicted_cost": float(row["predicted_cost"]),
                "predicted_co2": float(row["predicted_co2"]),
                "eco_score": float(row["eco_score"]),
            }
        )

    return jsonify(
        {
            "user_prediction": {
                "predicted_cost": round(predicted_cost, 2),
                "predicted_co2": round(predicted_co2, 2),
                "eco_score": round(eco_score, 3),
            },
            "top_recommendations": recommendations,
        }
    )


@app.route("/download")
def download_excel():
    df = get_recommendations_df().sort_values(by="eco_score", ascending=False).head(10)

    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    return send_file(
        output,
        download_name="eco_recommendations.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/health")
def health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)