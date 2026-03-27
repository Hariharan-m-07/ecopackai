from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sqlalchemy import create_engine
from flask import send_file
import io
app = Flask(__name__)

# -----------------------------
# DATABASE CONFIG
# -----------------------------
DB_USER = "postgres"
DB_PASSWORD = "mrbean"   # your password
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecopackai"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -----------------------------
# LOAD MODELS
# -----------------------------
cost_model = joblib.load("cost_model.pkl")
co2_model = joblib.load("co2_model.pkl")


# -----------------------------
# HOME ROUTE (UI)
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# RECOMMENDATION API
# -----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json

    # -------- INPUT --------
    material_type = data.get("material_type", "Bioplastic")
    strength = float(data["strength"])
    weight = float(data["weight"])
    biodeg = float(data["biodegradability"])
    recycle = float(data["recyclability"])

    # -------- USER INPUT DF --------
    input_data = pd.DataFrame({
        "material_type": [material_type],
        "strength_rating": [strength],
        "weight_capacity_kg": [weight],
        "biodegradability_score": [biodeg],
        "recyclability_percent": [recycle]
    })

    # -------- PREDICTION --------
    predicted_cost = float(cost_model.predict(input_data)[0])
    predicted_co2 = float(co2_model.predict(input_data)[0])

    eco_score = float((1/predicted_cost + 1/predicted_co2) / 2)

    # -----------------------------
    # LOAD FULL DATA
    # -----------------------------
    df = pd.read_sql("SELECT * FROM materials_cleaned", engine)

    # -------- PREDICT FOR ALL --------
    df["predicted_cost"] = cost_model.predict(df)
    df["predicted_co2"] = co2_model.predict(df)

    # -------- ECO SCORE --------
    df["eco_score"] = (1/df["predicted_cost"] + 1/df["predicted_co2"]) / 2

    # -------- TOP 5 --------
    top5 = df.sort_values(by="eco_score", ascending=False).head(5)

    # -------- CLEAN JSON --------
    recommendations = []
    for _, row in top5.iterrows():
        recommendations.append({
            "material_type": row["material_type"],
            "predicted_cost": float(row["predicted_cost"]),
            "predicted_co2": float(row["predicted_co2"]),
            "eco_score": float(row["eco_score"])
        })

    # -----------------------------
    # FINAL RESPONSE
    # -----------------------------
    return jsonify({
        "user_prediction": {
            "predicted_cost": round(predicted_cost, 2),
            "predicted_co2": round(predicted_co2, 2),
            "eco_score": round(eco_score, 3)
        },
        "top_recommendations": recommendations
    })

@app.route("/download")
def download_excel():

    # Load data
    df = pd.read_sql("SELECT * FROM materials_cleaned", engine)

    # Predictions
    df["predicted_cost"] = cost_model.predict(df)
    df["predicted_co2"] = co2_model.predict(df)

    # Eco score
    df["eco_score"] = (1/df["predicted_cost"] + 1/df["predicted_co2"]) / 2

    # Top 10
    top10 = df.sort_values(by="eco_score", ascending=False).head(10)

    # Convert to Excel in memory
    output = io.BytesIO()
    top10.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)

    return send_file(
        output,
        download_name="eco_recommendations.xlsx",
        as_attachment=True
    )
# -----------------------------
# RUN SERVER
# -----------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)