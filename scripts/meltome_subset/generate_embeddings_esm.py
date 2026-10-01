import os
from pathlib import Path
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

# === CONFIGURACIÓN ===
ROOT = Path(__file__).resolve().parents[2]
csv_path = ROOT / "data" / "meltome_subset_2335.csv"  # debe tener columnas: Protein, sequence
output_dir = ROOT / "embeddings"
esm_model_name = "facebook/esm2_t33_650M_UR50D"  # puedes cambiarlo por otro ESM2

# === CARGA DEL MODELO ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
model = EsmModel.from_pretrained(esm_model_name).to(device)
model.eval()

# === CARGA DEL DATASET ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=["sequence", "Protein"])

# === PROCESAMIENTO ===
os.makedirs(output_dir, exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df)):
    seq = row["sequence"]
    uid = row["Protein"]
    output_path = os.path.join(output_dir, f"{uid}.pkl")
    
    if os.path.exists(output_path):
        continue  # evitar recalcular si ya existe

    try:
        inputs = tokenizer(seq, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        with open(output_path, "wb") as f:
            pickle.dump(embedding, f)
    except Exception as e:
        print(f"❌ Error con {uid}: {e}")