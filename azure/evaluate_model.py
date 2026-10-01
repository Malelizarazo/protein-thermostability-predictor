import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

def evaluate_predictions(y_true, y_pred):
    """Evaluar el modelo usando múltiples métricas."""
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }
    return metrics

def plot_predictions(y_true, y_pred, save_path):
    """Generar gráfico de predicciones vs valores reales."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--', lw=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Azure Model - Predicciones vs Valores reales')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Crear directorios necesarios
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("graficos", exist_ok=True)

    print("Cargando datos...")
    # Cargar datos desde el archivo parquet
    df = pd.read_parquet("../intento_modelo_gpu/embeddings/embeddings.parquet")
    print("Forma del DataFrame:", df.shape)
    print("Columnas del DataFrame:", df.columns.tolist())

    # Separar características y objetivo
    X = df.drop(columns=["Protein_ID", "meltPoint", "meltPoint_normalized"])
    y = df["meltPoint"]

    print("Columnas en X:", X.columns.tolist())
    print("Forma de X:", X.shape)

    # Asegúrate de que X tenga las mismas columnas que el modelo espera
    # Cargar el modelo
    model = joblib.load('model.pkl')

    # Obtener columnas originales del entrenamiento si existe atributo 'feature_names_in_'
    if hasattr(model, 'feature_names_in_'):
        expected_columns = model.feature_names_in_.tolist()
    else:
        # Si estás usando un Pipeline, busca dentro del paso correspondiente
        try:
            expected_columns = model.named_steps['regressor'].feature_names_in_.tolist()
        except:
            expected_columns = X.columns.tolist()  # fallback (NO recomendado)

    # Reordenar columnas
    X = X.reindex(columns=expected_columns, fill_value=0)

    # Verificar diferencia entre columnas del modelo y del DataFrame actual
    missing_cols = list(set(expected_columns) - set(X.columns))
    extra_cols = list(set(X.columns) - set(expected_columns))

    print(f"\n🟠 Columnas faltantes (esperadas por el modelo pero no presentes en X): {missing_cols}")
    print(f"🔵 Columnas extras (presentes en X pero no esperadas por el modelo): {extra_cols}")
    print(f"✅ Modelo espera {len(expected_columns)} columnas — X tiene {X.shape[1]}")

    print("Cargando modelo...")
    # Realizar predicciones
    y_pred = model.predict(X)
    
    # Calcular métricas
    metrics = evaluate_predictions(y, y_pred)
    
    # Imprimir resultados
    print("\nResultados de la evaluación:")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Explained Variance: {metrics['explained_variance']:.4f}")

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar métricas
    results_file = f'resultados/azure_model_evaluation_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResultados guardados en: {results_file}")

    # Generar y guardar gráfico
    plot_predictions(
        y, 
        y_pred, 
        f'graficos/azure_predictions_{timestamp}.png'
    )
    print(f"Gráfico guardado en: graficos/azure_predictions_{timestamp}.png")

if __name__ == "__main__":
    main() 