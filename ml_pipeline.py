import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
print("ML Pipeline Started...")

# -----------------------------
# Database Connection
# -----------------------------
DB_USER = "postgres"
DB_PASSWORD = "mrbean"  # CHANGE
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecopackai"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
df = pd.read_sql("SELECT * FROM materials_cleaned;", engine)
print("Dataset Loaded:", df.shape)

# -----------------------------
# STEP 2: Select Features
# -----------------------------
features = [
    "material_type",
    "strength_rating",
    "weight_capacity_kg",
    "biodegradability_score",
    "recyclability_percent",
    "co2_impact_index",
    "cost_efficiency_index",
    "material_suitability_score",
]
X = df[features]

# -----------------------------
# STEP 3: Create Targets
# -----------------------------
y_cost = df["cost_per_unit"]
y_co2 = df["co2_emission_score"]

# -----------------------------
# STEP 4: Encode Categorical
# -----------------------------
categorical_features = ["material_type"]
numeric_features = [
    "strength_rating",
    "weight_capacity_kg",
    "biodegradability_score",
    "recyclability_percent",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# -----------------------------
# STEP 5: Train/Test Split
# -----------------------------
X_train, X_test, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)

_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=0.2, random_state=42
)

print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)

# -----------------------------
# STEP 7: Random Forest (Cost)
# -----------------------------
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42))
    ]
)

rf_pipeline.fit(X_train, y_cost_train)
joblib.dump(rf_pipeline, "cost_model.pkl")
# -----------------------------
# Compute feature importance for cost model
# -----------------------------
# Get feature names from pipeline preprocessor
# numeric_features defined earlier; expand categorical via OneHotEncoder
try:
    cat_features = list(
        rf_pipeline.named_steps["preprocessor"].
        named_transformers_["cat"].get_feature_names_out(categorical_features)
    )
except Exception:
    # fallback if transformer not accessible
    cat_features = []

feature_names = numeric_features + cat_features
importances = rf_pipeline.named_steps["model"].feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances,
}).sort_values(by="Importance", ascending=False)


# make predictions for cost

y_cost_pred = rf_pipeline.predict(X_test)

# -----------------------------
# STEP 8: XGBoost (CO2)
# -----------------------------
xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ]
)

xgb_pipeline.fit(X_train, y_co2_train)
joblib.dump(xgb_pipeline, "co2_model.pkl")
y_co2_pred = xgb_pipeline.predict(X_test)

# -----------------------------
# STEP 9: Evaluation
# -----------------------------
print("\n--- Cost Prediction Metrics ---")
print("RMSE:", np.sqrt(mean_squared_error(y_cost_test, y_cost_pred)))
print("MAE:", mean_absolute_error(y_cost_test, y_cost_pred))
print("R2:", r2_score(y_cost_test, y_cost_pred))

print("\n--- CO2 Prediction Metrics ---")
print("RMSE:", np.sqrt(mean_squared_error(y_co2_test, y_co2_pred)))
print("MAE:", mean_absolute_error(y_co2_test, y_co2_pred))
print("R2:", r2_score(y_co2_test, y_co2_pred))

# -----------------------------
# STEP 10: Ranking System
# -----------------------------
df_test = X_test.copy()
df_test["predicted_cost"] = y_cost_pred
df_test["predicted_co2"] = y_co2_pred

df_test["eco_score"] = (
    0.5 * (1 / df_test["predicted_cost"]) +
    0.5 * (1 / df_test["predicted_co2"])
)

df_ranked = df_test.sort_values(by="eco_score", ascending=False)

print("\nTop 5 Recommended Materials:")
print(df_ranked.head())
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance (Cost Model)")
plt.gca().invert_yaxis()
plt.show()