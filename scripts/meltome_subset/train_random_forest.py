import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# === Rutas ===
ROOT = Path(__file__).resolve().parents[2]
csv_path = ROOT / "data" / "meltome_subset_2335.csv"
embedding_dir = ROOT / "embeddings"
modelo_path = ROOT / "models" / "rf_model.pkl"

# === Cargar CSV ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=["Protein", "meltPoint"])

# === Cargar embeddings ===
X = []
y = []
ids = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    uid = row["Protein"]
    label = row["meltPoint"]
    emb_path = os.path.join(embedding_dir, f"{uid}.pkl")
    if not os.path.exists(emb_path):
        continue
    try:
        with open(emb_path, "rb") as f:
            emb = pickle.load(f)
        X.append(emb)
        y.append(label)
        ids.append(uid)
    except Exception as e:
        print(f"❌ Error con {uid}: {e}")

X = np.array(X)
y = np.array(y)

# === Dividir datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Entrenar modelo solo con train ===
print("Entrenando modelo Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === Evaluar en test ===
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
pcc, _ = pearsonr(y_test, y_pred)

print("\n=== Evaluación en conjunto de test ===")
print(f"✅ R²: {r2:.4f}")
print(f"✅ MAE: {mae:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ PCC: {pcc:.4f}")

# === Guardar modelo (scaler + RF juntos, para que la inferencia use el mismo escalado) ===
pipeline = Pipeline([("scaler", scaler), ("rf", model)])
joblib.dump(pipeline, modelo_path)
print(f"📦 Modelo guardado en: {modelo_path}")
