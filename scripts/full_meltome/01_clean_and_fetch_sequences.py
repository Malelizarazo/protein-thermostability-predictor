import pandas as pd
import requests
import time
from tqdm import tqdm
import random
import os
import h5py
import torch

def fetch_sequence_individual(protein_id):
    """Fetches amino acid sequence for a single UniProt ID."""
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            sequence = ''.join(lines[1:])  # Join the sequence lines
            return sequence
        else:
            print(f"Error: Received status code {response.status_code} for ID: {protein_id}")
    except Exception as e:
        print(f"Error fetching sequence for {protein_id}: {e}")
    return None  # Return None if there's an error

def test_random_uniprot_ids(df):
    """Fetches sequences for a few random UniProt IDs to test the API."""
    random_ids = random.sample(list(df['Protein_ID']), min(10, len(df)))  # Get up to 10 random IDs
    print("Testing with random UniProt IDs:", random_ids)
    
    for protein_id in random_ids:
        sequence = fetch_sequence_individual(protein_id)
        print(f"Secuencia obtenida para {protein_id}:", sequence)

def fetch_sequences(protein_ids):
    """Fetches amino acid sequences for a list of UniProt IDs."""
    base_url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query="
    ids_str = ' OR '.join([f"id:{pid}" for pid in protein_ids])  # Format for batch query
    url = f"{base_url}{ids_str}"  # Construct the URL
    print(f"Fetching from URL: {url}")  # Debugging line
    try:
        response = requests.get(url)
        if response.status_code == 200:
            fasta_entries = response.text.strip().split('>')[1:]  # Ignore the first empty entry
            sequences = {}
            for entry in fasta_entries:
                lines = entry.split('\n')
                uniprot_id = lines[0].split()[0]  # Get the UniProt ID
                sequence = ''.join(lines[1:])  # Join the sequence lines
                sequences[uniprot_id] = sequence
            return sequences
        else:
            print(f"Error: Received status code {response.status_code} for IDs: {protein_ids}")
    except Exception as e:
        print(f"Error fetching sequences for {protein_ids}: {e}")
    return {pid: None for pid in protein_ids}  # Return None for all if there's an error

def get_sequences_from_uniprot(uniprot_ids):
    """Obtiene las secuencias de aminoácidos de UniProt en lotes."""
    sequences = {}
    batch_size = 100  # Increased batch size for efficiency
    sleep_time = 1    # Pause between batches in seconds

    # Use tqdm to create a progress bar for the batches
    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="Obteniendo secuencias de UniProt"):
        batch_ids = uniprot_ids[i:i + batch_size]
        
        # Fetch sequences for the current batch
        batch_sequences = fetch_sequences(batch_ids)
        sequences.update(batch_sequences)  # Update the main dictionary with the results

        time.sleep(sleep_time)  # To avoid overloading the API
    return sequences

def limpiar_datos(input_file, output_file, log_file):
    """Limpia los datos según los criterios especificados."""
    # Cargar datos
    print(f"Cargando datos desde {input_file}...")
    df = pd.read_csv(input_file, dtype={'Protein_ID': str, 'meltPoint': float})
    
    # Imprimir nombres de columnas
    print("Nombres de columnas:", df.columns.tolist())
    
    print(f"Datos originales: {len(df)} filas")
    
    # Filtrar filas con meltPoint
    df = df.dropna(subset=['meltPoint'])
    print(f"Después de filtrar por meltPoint: {len(df)} filas")
    
    # Extraer solo la parte del ID de UniProt antes del guion bajo
    df['Protein_ID'] = df['Protein_ID'].str.split('_').str[0]
    
    # Asegurarse de que todos los IDs sean cadenas
    df['Protein_ID'] = df['Protein_ID'].astype(str)
    
    # Filtrar cualquier ID que sea nulo o vacío
    df = df[df['Protein_ID'].str.strip() != '']
    print(f"Después de filtrar IDs vacíos: {len(df)} filas")
    
    # Eliminar filas donde Protein_ID es nulo
    df = df[df['Protein_ID'].notnull()]
    
    # Eliminar duplicados basados en UniProt ID
    df = df.drop_duplicates(subset=['Protein_ID'])
    
    # Open log file
    with open(log_file, 'w') as log:
        # Obtener secuencias de UniProt
        for protein_id in tqdm(df['Protein_ID'], desc="Obteniendo secuencias de UniProt"):
            sequence = fetch_sequence_individual(protein_id)
            if sequence:
                log.write(f"{protein_id}: {sequence}\n")  # Log the sequence
            else:
                log.write(f"{protein_id}: No sequence found\n")  # Log if no sequence found
            time.sleep(1)  # Pause to avoid overloading the API

    # Save the DataFrame with sequences
    df['sequence'] = df['Protein_ID'].map(lambda pid: fetch_sequence_individual(pid))
    df = df.dropna(subset=['sequence'])
    df.to_csv(output_file, index=False)
    print(f"Datos guardados en {output_file}")
    print(f"Número final de proteínas: {len(df)}")

def main():
    # Crear directorio de embeddings si no existe
    os.makedirs("../../embeddings", exist_ok=True)
    
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
    embeddings = generator.generate_embeddings(df['sequence'].tolist())
    
    # Guardar embeddings en formato HDF5
    print("Guardando embeddings...")
    with h5py.File("../../embeddings/embeddings.h5", "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("ids", data=df['Protein_ID'].astype(str))  # Updated to 'Protein_ID'
        f.create_dataset("meltpoints", data=df['meltPoint'])
    
    print("Embeddings guardados exitosamente")
    print(f"Forma de los embeddings: {len(embeddings)} x {len(embeddings[0])}")

if __name__ == "__main__":
    input_file = "../../data/original.csv"
    output_file = "../../data/datos_limpios.csv"
    log_file = "../../data/log.txt"  # Specify the log file path
    limpiar_datos(input_file, output_file, log_file)
    # Los embeddings se generan después con 02_generate_embeddings.py 