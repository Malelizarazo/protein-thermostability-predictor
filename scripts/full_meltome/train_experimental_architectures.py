import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

# Modelos adicionales
class CNNRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        # Calcular el tamaño de salida después de CNN
        cnn_output = 32 * (input_size // 4)
        self.fc = nn.Sequential(
            nn.Linear(cnn_output, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Agregar dimensión de canal
        x = self.cnn(x)
        return self.fc(x)

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, nhead=8):
        super().__init__()
        self.embedding = nn.Linear(input_size, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=nhead, dim_feedforward=2048, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove sequence dimension
        return self.fc(x)

class HybridMLPRegressor(nn.Module):
    def __init__(self, input_size, n_clusters):
        super().__init__()
        self.cluster_embedding = nn.Embedding(n_clusters, 32)
        self.network = nn.Sequential(
            nn.Linear(input_size + 32, 512),  # +32 for cluster embedding
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, cluster_ids):
        if cluster_ids.dim() == 1:
            cluster_ids = cluster_ids.unsqueeze(1)  # Asegura que sea 2D
        cluster_emb = self.cluster_embedding(cluster_ids).squeeze(1)  # [batch_size, 32]
        x = torch.cat([x, cluster_emb], dim=1)
        return self.network(x)

def train_model_with_clusters(model, train_loader, val_loader, criterion, optimizer, device, model_name, epochs=100):
    best_val_loss = float('inf')
    results = []
    
    for epoch in tqdm(range(epochs), desc=f"Entrenando {model_name}"):
        # Entrenamiento
        model.train()
        train_loss = 0
        for X, y, cluster_id in train_loader:
            X, y = X.to(device), y.to(device)
            cluster_id = cluster_id.to(device)
            optimizer.zero_grad()
            output = model(X, cluster_id)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validación
        model.eval()
        val_loss = 0
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for X, y, cluster_id in val_loader:
                X, y = X.to(device), y.to(device)
                cluster_id = cluster_id.to(device)
                output = model(X, cluster_id)
                val_loss += criterion(output.squeeze(), y).item()
                predictions.extend(output.squeeze().cpu().numpy())
                true_values.extend(y.cpu().numpy())
        
        # Calcular métricas
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # Guardar resultados
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'../../models/{model_name}_best.pt')
    
    return results

def evaluate_model_with_clusters(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for X, y, cluster_id in test_loader:
            X, y = X.to(device), y.to(device)
            cluster_id = cluster_id.to(device)
            output = model(X, cluster_id)
            test_loss += criterion(output.squeeze(), y).item()
            predictions.extend(output.squeeze().cpu().numpy())
            true_values.extend(y.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    return {
        'test_loss': test_loss,
        'r2_score': r2_score(true_values, predictions),
        'mse': mean_squared_error(true_values, predictions),
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mae': mean_absolute_error(true_values, predictions),
        'explained_variance': explained_variance_score(true_values, predictions),
        'predictions': predictions,
        'true_values': true_values
    }

def plot_cluster_distribution(embeddings, cluster_labels, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(cluster_labels, bins=len(np.unique(cluster_labels)))
    plt.title('Distribución de Clusters')
    plt.xlabel('Cluster ID')
    plt.ylabel('Cantidad de muestras')
    plt.savefig(save_path)
    plt.close()

def main():
    # Crear directorios necesarios
    os.makedirs("../../models", exist_ok=True)
    os.makedirs("../../results", exist_ok=True)
    os.makedirs("../../results/plots", exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar embeddings
    print("Cargando embeddings...")
    with h5py.File("../../embeddings/embeddings.h5", "r") as f:
        embeddings = f["embeddings"][:]
        protein_ids = f["ids"][:]
        meltpoints = f["meltpoints"][:]

    # Convertir protein_ids a str para mapear correctamente
    protein_ids = [pid.decode('utf-8').strip() for pid in protein_ids]
    meltpoint_dict = dict(zip(protein_ids, meltpoints))

    # Asegurarse de tomar solo los meltpoints correspondientes a los embeddings
    filtered_meltpoints = np.array([meltpoint_dict[pid] for pid in protein_ids])

    # Normalizar datos
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Aplicar clustering
    n_clusters = 5  # Puedes ajustar este número
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    # Guardar distribución de clusters
    plot_cluster_distribution(
        embeddings_scaled, 
        cluster_labels, 
        f'../../results/plots/cluster_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    
    # Split de datos
    X_train_val, X_test, y_train_val, y_test, clusters_train_val, clusters_test = train_test_split(
        embeddings, filtered_meltpoints, cluster_labels, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, clusters_train, clusters_val = train_test_split(
        X_train_val, y_train_val, clusters_train_val, test_size=0.2, random_state=42
    )
    
    # Convertir a tensores
    train_data = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train),
        torch.LongTensor(clusters_train)
    )
    val_data = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val),
        torch.LongTensor(clusters_val)
    )
    test_data = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test),
        torch.LongTensor(clusters_test)
    )
    
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Definir modelos
    input_size = embeddings.shape[1]
    models = {
        'HybridMLP': HybridMLPRegressor(input_size, n_clusters),
        'CNN': CNNRegressor(input_size),
        'Transformer': TransformerRegressor(input_size)
    }
    
    # Entrenar y evaluar cada modelo
    for model_name, model in models.items():
        print(f"\nEntrenando modelo: {model_name}")
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        if model_name == 'HybridMLP':
            results = train_model_with_clusters(
                model, train_loader, val_loader, criterion, optimizer, device, model_name
            )
            eval_results = evaluate_model_with_clusters(
                model, test_loader, criterion, device
            )
        else:
            # Implementar entrenamiento estándar para CNN y Transformer
            pass  # TODO: Implementar entrenamiento estándar
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'../../results/{model_name}_evaluation_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump({
                'test_loss': eval_results['test_loss'],
                'r2_score': eval_results['r2_score'],
                'mse': eval_results['mse'],
                'rmse': eval_results['rmse'],
                'mae': eval_results['mae'],
                'explained_variance': eval_results['explained_variance']
            }, f)
        
        print(f"Resultados guardados en: {results_file}")
        
        # Generar gráficos
        plt.figure(figsize=(10, 6))
        plt.scatter(eval_results['true_values'], eval_results['predictions'], alpha=0.5)
        plt.plot([min(eval_results['true_values']), max(eval_results['true_values'])],
                [min(eval_results['true_values']), max(eval_results['true_values'])],
                'r--', lw=2)
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title(f'{model_name} - Predicciones vs Valores reales')
        plt.savefig(f'../../results/plots/{model_name}_predictions_{timestamp}.png')
        plt.close()

if __name__ == "__main__":
    main()
