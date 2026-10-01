import torch
from transformers import EsmTokenizer, EsmModel
import pandas as pd
import h5py
from tqdm import tqdm
import os
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", token=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        print(f"Cargando modelo {model_name}...")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name, token=token)
        self.model = EsmModel.from_pretrained(model_name, token=token).to(self.device)
        self.model.eval()

    def generate_embeddings(self, sequences, protein_ids, batch_size=10):
        embeddings = []
        valid_protein_ids = []  # To store IDs of successfully processed sequences
        
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i:i + batch_size]
            batch_ids = protein_ids[i:i + batch_size]  # Get corresponding protein IDs
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Process each sequence individually
            for j in range(len(batch)):
                try:
                    with torch.no_grad():
                        # Process one sequence at a time
                        output = self.model(**{k: v[j:j+1] for k, v in inputs.items()})
                        embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()
                        embeddings.append(embedding.squeeze(0))  # Remove the extra dimension
                        valid_protein_ids.append(batch_ids[j])  # Store the corresponding protein ID
                except Exception as e:
                    print(f"Error processing sequence: {batch[j]}")  # Log the sequence that caused the error
                    print(f"Error message: {e}")
                    continue  # Skip this sequence and continue with the next one
        
        return embeddings, valid_protein_ids  # Return both embeddings and valid protein IDs

def main():
    # Crear directorio de embeddings si no existe
    os.makedirs("../../embeddings", exist_ok=True)
    
    # Eliminar el archivo de embeddings si existe
    embeddings_file_path = "../../embeddings/embeddings.h5"
    if os.path.exists(embeddings_file_path):
        os.remove(embeddings_file_path)
        print(f"Archivo existente {embeddings_file_path} eliminado.")

    # Cargar datos limpios
    print("Cargando datos limpios...")
    df = pd.read_csv("../../data/datos_limpios.csv")
    
    # Filtrar secuencias nulas o vacías
    df = df[df['sequence'].notnull() & (df['sequence'] != "")]
    print(f"Datos cargados: {len(df)} proteínas válidas")
    
    # Inicializar generador de embeddings con el token
    token = os.environ.get("HF_TOKEN")  # opcional: ESM-2 es un modelo público
    generator = EmbeddingGenerator(token=token)
    
    # Generar embeddings
    print("Generando embeddings...")
    embeddings, valid_protein_ids = generator.generate_embeddings(df['sequence'].tolist(), df['Protein_ID'].tolist())
    
    # Guardar embeddings en formato HDF5
    print("Guardando embeddings...")

    # Asegurarte de usar meltpoints que correspondan a las secuencias válidas
    meltpoints_array = df.loc[df['sequence'].notnull() & (df['sequence'] != ""), 'meltPoint'].to_numpy()

    # Convertir protein IDs a string binario fijo (S10)
    protein_ids_array = np.array(valid_protein_ids, dtype='S10')

    with h5py.File(embeddings_file_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("ids", data=protein_ids_array)
        f.create_dataset("meltpoints", data=meltpoints_array)
    
    print("Embeddings guardados exitosamente")
    print(f"Forma de los embeddings: {np.array(embeddings).shape}")  # Should be (N, 1280)

if __name__ == "__main__":
    main() 