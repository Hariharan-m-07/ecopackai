import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine

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

sns.set(style="whitegrid")

# -----------------------------
# Load Data
# -----------------------------
raw_df = pd.read_sql("SELECT * FROM materials;", engine)
clean_df = pd.read_sql("SELECT * FROM materials_cleaned;", engine)

print("Raw Shape:", raw_df.shape)
print("Clean Shape:", clean_df.shape)

# -----------------------------
# 1️⃣ Material Type Distribution
# -----------------------------
plt.figure(figsize=(8,5))
raw_df["material_type"].value_counts().plot(kind="bar")
plt.title("Material Type Distribution (Raw Dataset)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 2️⃣ Missing Values Heatmap (Raw)
# -----------------------------
plt.figure(figsize=(8,5))
sns.heatmap(raw_df.isnull(), cbar=False)
plt.title("Missing Values Heatmap (Raw Dataset)")
plt.tight_layout()
plt.show()

# -----------------------------
# 3️⃣ Boxplot for Outliers (Before Cleaning)
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=raw_df[["cost_per_unit", "co2_emission_score"]])
plt.title("Outlier Visualization (Raw Dataset)")
plt.tight_layout()
plt.show()

# -----------------------------
# 4️⃣ Sustainability Scores Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(clean_df["material_suitability_score"], kde=True)
plt.title("Material Suitability Score Distribution")
plt.tight_layout()
plt.show()

# -----------------------------
# 5️⃣ Correlation Heatmap (Cleaned Data)
# -----------------------------
plt.figure(figsize=(10,6))
corr = clean_df.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap (Cleaned Dataset)")
plt.tight_layout()
plt.show()

# -----------------------------
# 6️⃣ Interactive Plotly Visualization
# -----------------------------
fig = px.scatter(
    clean_df,
    x="cost_per_unit",
    y="co2_impact_index",
    color="material_type",
    title="Cost vs CO2 Impact (Interactive)"
)

fig.show()
