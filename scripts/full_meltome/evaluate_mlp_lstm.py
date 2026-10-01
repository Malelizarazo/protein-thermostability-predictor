import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from tqdm import tqdm

class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        # Ensure embeddings are properly shaped
        # If embeddings are 3D (batch, sequence_length, embedding_dim), reshape to 2D (batch, embedding_dim)
        if len(embeddings.shape) == 3:
            # Take the mean across the sequence length dimension
            embeddings = np.mean(embeddings, axis=1)
        
        # Convert to torch tensors
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class MLPRegressor(nn.Module):
    def __init__(self, input_size=1280):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            test_loss += criterion(output.squeeze(), y).item()
            predictions.extend(output.squeeze().cpu().numpy())
            true_values.extend(y.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    
    # Calculate metrics
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    r2 = r2_score(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    explained_var = explained_variance_score(true_values, predictions)
    
    return {
        'test_loss': test_loss,
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'explained_variance': explained_var,
        'predictions': predictions,
        'true_values': true_values
    }

def plot_predictions_vs_actual(predictions, true_values, model_name, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Predictions vs Actual Values')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(predictions, true_values, model_name, save_path):
    residuals = predictions - true_values
    plt.figure(figsize=(10, 8))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residual Plot')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(predictions, true_values, model_name, save_path):
    errors = predictions - true_values
    plt.figure(figsize=(10, 8))
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f'{model_name} - Error Distribution')
    plt.tight_layout()
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

    # Print the shape of the embeddings to debug
    print("Shape of embeddings:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])

    # Convert protein_ids from bytes to strings for easier comparison
    protein_ids = [pid.decode('utf-8').strip() for pid in protein_ids]

    # Create a mapping of protein IDs to meltpoints
    meltpoint_dict = {pid: mp for pid, mp in zip(protein_ids, meltpoints)}

    # Filter meltpoints to only include those corresponding to the protein IDs that have embeddings
    # This ensures we only use meltpoints for proteins that have embeddings
    filtered_meltpoints = [meltpoint_dict[pid] for pid in protein_ids]

    # Check lengths
    print("Length of embeddings:", len(embeddings))
    print("Length of filtered meltpoints:", len(filtered_meltpoints))

    # Ensure the lengths match before proceeding
    if len(embeddings) != len(filtered_meltpoints):
        raise ValueError("The number of embeddings and filtered meltpoints do not match.")

    # Proceed with train_test_split - first split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        embeddings, filtered_meltpoints, test_size=0.2, random_state=42
    )
    
    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Preparar dataloaders
    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    test_dataset = ProteinDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Definir modelos a entrenar
    # Use the actual embedding dimension from the loaded data
    embedding_dim = embeddings.shape[1]
    print(f"Using embedding dimension: {embedding_dim}")
    
    models = {
        'MLP': MLPRegressor(input_size=embedding_dim),
        'LSTM': LSTMRegressor(input_size=embedding_dim)
    }
    
    # Cargar los mejores modelos entrenados previamente
    for model_name, model in models.items():
        model_path = f'../../models/{model_name}_best.pt'
        if os.path.exists(model_path):
            print(f"Cargando modelo {model_name} desde {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"No se encontró el modelo {model_name} en {model_path}")
            continue
        
        model = model.to(device)
        criterion = nn.MSELoss()
        
        # Evaluar el modelo en el conjunto de prueba
        print(f"\nEvaluando modelo: {model_name}")
        results = evaluate_model(model, test_loader, criterion, device)
        
        # Imprimir resultados
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"R² Score: {results['r2_score']:.4f}")
        print(f"MSE: {results['mse']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"Explained Variance: {results['explained_variance']:.4f}")
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'../../results/{model_name}_evaluation_{timestamp}.json'
        
        # Convertir arrays a listas para JSON
        results_for_json = {
            'test_loss': results['test_loss'],
            'r2_score': results['r2_score'],
            'mse': results['mse'],
            'rmse': results['rmse'],
            'mae': results['mae'],
            'explained_variance': results['explained_variance']
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_for_json, f)
        print(f"Resultados guardados en: {results_file}")
        
        # Generar gráficos
        plot_predictions_vs_actual(
            results['predictions'], 
            results['true_values'], 
            model_name, 
            f'../../results/plots/{model_name}_predictions_vs_actual_{timestamp}.png'
        )
        
        plot_residuals(
            results['predictions'], 
            results['true_values'], 
            model_name, 
            f'../../results/plots/{model_name}_residuals_{timestamp}.png'
        )
        
        plot_error_distribution(
            results['predictions'], 
            results['true_values'], 
            model_name, 
            f'../../results/plots/{model_name}_error_distribution_{timestamp}.png'
        )
        
        print(f"Gráficos guardados en: ../../results/plots/")

if __name__ == "__main__":
    main()
