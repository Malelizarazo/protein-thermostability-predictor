import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions(y_true, y_pred):
    """Evaluar el modelo usando múltiples métricas."""
    return {
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }

def plot_predictions(y_true, y_pred, save_path):
    """Generar gráfico de predicciones vs valores reales."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--', lw=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('ElasticNet - Predicciones vs Valores reales')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Crear directorios necesarios
    os.makedirs("../../models", exist_ok=True)
    os.makedirs("../../results", exist_ok=True)
    os.makedirs("../../results/plots", exist_ok=True)

    print("Cargando datos...")
    # Cargar datos desde el archivo parquet
    df = pd.read_parquet("../../embeddings/embeddings.parquet")

    # Separar características y objetivo
    X = df.drop(columns=["Protein_ID", "meltPoint", "meltPoint_normalized"])
    y = df["meltPoint"]

    print("Dividiendo datos en entrenamiento y prueba...")
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Creando y entrenando el modelo...")
    # Crear pipeline con los parámetros específicos
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False, with_std=True)),
        ("regressor", ElasticNet(alpha=0.001, l1_ratio=0.8436842105263158))
    ])

    # Entrenar modelo
    pipeline.fit(X_train, y_train)

    # Realizar predicciones
    print("Evaluando el modelo...")
    y_pred = pipeline.predict(X_test)
    
    # Calcular métricas
    metrics = evaluate_predictions(y_test, y_pred)
    
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
    results_file = f'../../results/elasticnet_evaluation_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f)
    print(f"\nResultados guardados en: {results_file}")

    # Generar y guardar gráfico
    plot_predictions(
        y_test, 
        y_pred, 
        f'../../results/plots/elasticnet_predictions_{timestamp}.png'
    )
    print(f"Gráfico guardado en: ../../results/plots/elasticnet_predictions_{timestamp}.png")

    # Guardar modelo
    model_path = f'../../models/elasticnet_model_{timestamp}.pkl'
    joblib.dump(pipeline, model_path)
    print(f"\nModelo guardado en: {model_path}")

if __name__ == "__main__":
    main()
