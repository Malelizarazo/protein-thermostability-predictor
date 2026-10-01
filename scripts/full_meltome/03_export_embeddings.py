import h5py
import pandas as pd
import numpy as np

# Ruta al archivo .h5
h5_path = "../../embeddings/embeddings.h5"

# Leer datos del archivo .h5
with h5py.File(h5_path, "r") as f:
    embeddings = f["embeddings"][:]
    ids = f["ids"][:].astype(str)  # Convertir de bytes a str
    all_ids = f["ids"][:].astype(str)
    all_meltpoints = f["meltpoints"][:]

# Crear diccionario de ID -> meltPoint
meltpoint_dict = dict(zip(all_ids, all_meltpoints))

# Obtener solo los meltPoints correspondientes a los embeddings válidos
matched_meltpoints = np.array([meltpoint_dict[pid] for pid in ids])

# Normalización Min-Max
min_val = matched_meltpoints.min()
max_val = matched_meltpoints.max()
normalized_meltpoints = (matched_meltpoints - min_val) / (max_val - min_val)

# Construir el DataFrame final
df = pd.DataFrame(embeddings)
df["Protein_ID"] = ids
df["meltPoint"] = matched_meltpoints
df["meltPoint_normalized"] = normalized_meltpoints

# Guardar como CSV
df.to_csv("../../embeddings/embeddings.csv", index=False)
print("✅ Archivo CSV guardado en ../../embeddings/embeddings.csv")

# Guardar como Parquet (requiere pyarrow o fastparquet)
try:
    df.to_parquet("../../embeddings/embeddings.parquet", index=False)
    print("✅ Archivo Parquet guardado en ../../embeddings/embeddings.parquet")
except ImportError:
    print("⚠️ Instala 'pyarrow' o 'fastparquet' para guardar en formato Parquet.")
