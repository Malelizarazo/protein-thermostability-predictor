# Evalúa el modelo guardado sobre TODO el dataset (incluye datos de entrenamiento).
# Las métricas sobre el conjunto de test (20%) las imprime train_random_forest.py.
import joblib
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

# 1. Cargar el pipeline (scaler + RF) guardado por train_random_forest.py
with open(ROOT / 'models' / 'rf_model.pkl', 'rb') as file:
    model = joblib.load(file)

# 2. Cargar y preparar los datos
data = pd.read_csv(ROOT / 'data' / 'meltome_subset_2335.csv')

# Verificar las columnas del archivo CSV
print(data.columns)

# 3. Cargar los embeddings
X = []
y = []

for _, row in data.iterrows():
    uid = row['Protein']
    emb_path = ROOT / 'embeddings' / f"{uid}.pkl"
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            emb = pickle.load(f)
        X.append(emb)
        y.append(row['meltPoint'])

# Convertir X a una matriz 2D
X = np.array(X)
y = np.array(y)

# Verificar que X no esté vacío
if X.size == 0:
    raise ValueError("No se encontraron embeddings. Asegúrate de que los archivos .pkl estén en la carpeta 'embeddings'.")

# 4. Realizar predicciones
predictions = model.predict(X)

# 5. Evaluar el modelo
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Coeficiente de Determinación (R²): {r2:.2f}')