import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import os
from datetime import datetime

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_name, epochs=100):
    best_val_loss = float('inf')
    results = []
    
    for epoch in tqdm(range(epochs), desc=f"Entrenando {model_name}"):
        # Entrenamiento
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
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
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
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

def main():
    # Crear directorios necesarios
    os.makedirs("../../models", exist_ok=True)
    os.makedirs("../../results", exist_ok=True)
    
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

    # Proceed with train_test_split
    X_train, X_val, y_train, y_val = train_test_split(embeddings, filtered_meltpoints, test_size=0.2, random_state=42)
    
    # Preparar dataloaders
    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Definir modelos a entrenar
    # Use the actual embedding dimension from the loaded data
    embedding_dim = embeddings.shape[1]
    print(f"Using embedding dimension: {embedding_dim}")
    
    models = {
        'MLP': MLPRegressor(input_size=embedding_dim),
        'LSTM': LSTMRegressor(input_size=embedding_dim)
    }
    
    # Entrenar cada modelo
    for model_name, model in models.items():
        print(f"\nEntrenando modelo: {model_name}")
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        results = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, model_name
        )
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'../../results/{model_name}_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f)
        print(f"Resultados guardados en: {results_file}")

    print("Shape of embeddings:", np.array(embeddings).shape)

if __name__ == "__main__":
    main() 