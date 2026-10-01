# Protein Thermostability Predictor

Predicts the melting temperature (Tm) of a protein from its amino acid sequence, using ESM-2 protein-language-model
embeddings and machine learning regression models. Includes a small web app for single-sequence inference.

This is my undergraduate thesis project (Chemical Engineering, Universidad de los Andes, 2025).

## Why

Thermal stability decides whether a protein keeps working at the temperatures used in industrial biocatalysis,
pharmaceutical formulation and food processing. Measuring Tm experimentally is slow and needs specialised
equipment, and physics-based simulations are computationally expensive. The goal was to estimate Tm from
sequence alone and make the prediction usable by people without an ML background.

## How it works

```
amino acid sequence
   └─> ESM-2 (facebook/esm2_t33_650M_UR50D) ─> mean-pooled 1,280-dim embedding
          └─> regression model ─> predicted Tm (°C)
```

The project went through three modelling iterations:

1. **ProThermDB (~11,000 entries):** compared ESM-2 and ProtTrans embeddings; ESM-2 won.
2. **Meltome Atlas subset (2,335 wild-type proteins):** compared Random Forest, Bayesian Ridge, KNN, SVR and MLP.
3. **Full Meltome Atlas wild-type set (20,000+ sequences, fetched from the UniProt API):** trained MLP and LSTM
   networks in PyTorch on GPU. The LSTM was deployed in the web app.

A model trained with Azure Machine Learning AutoML is also included in `azure/`.

## Results

From the thesis report (80/20 train/validation split). Full tables are in [docs/results.md](docs/results.md).

| Iteration | Data | Best model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|---|---|
| 2 | Meltome subset, 2,335 proteins | Random Forest | 6.12 | 8.57 | 0.81 |
| 3 | Meltome wild-type, 20,000+ sequences | LSTM | 4.54 | 5.88 | 0.74 |

The two iterations use different datasets, so their numbers are not directly comparable. Both neural models
under-predict the hottest proteins (the LSTM saturates around 85 °C).

## Stack

Python · PyTorch · Hugging Face Transformers (ESM-2) · scikit-learn · pandas · FastAPI · Azure Machine Learning

## Repository layout

```
app/        FastAPI web app: sequence in, predicted Tm out (LSTM model)
models/     lstm_best.pt, trained LSTM weights used by the app
scripts/    embedding generation, Random Forest training and evaluation (iteration 2)
data/       meltome_subset_2335.csv: UniProt ID, Tm and sequence for the iteration-2 subset
azure/      Azure ML AutoML model, scoring script, environment and evaluation script
docs/       results.md: metrics for every iteration
```

The training code for iteration 3 (full Meltome set, MLP/LSTM) is not part of this repository. The trained
LSTM weights are.

## Run the web app

```bash
pip install -r app/requirements.txt
cd app
uvicorn app:app --reload
```

Open http://localhost:8000 and paste a sequence. The first run downloads ESM-2 from Hugging Face (about 2.5 GB).
A GPU is used automatically if available.

## Reproduce the Random Forest results (iteration 2)

```bash
pip install -r requirements.txt
python scripts/generate_embeddings_esm.py   # writes one ESM-2 embedding per protein to embeddings/
python scripts/train_random_forest.py       # 80/20 split, prints test MAE / RMSE / R² / PCC, saves models/rf_model.pkl
```

## Data sources

- Jarząb et al., "Meltome atlas — thermal proteome stability across the tree of life", *Nature Methods* 17, 2020.
  doi:10.1038/s41592-020-0801-4
- Kumar et al., "ProTherm and ProNIT: thermodynamic databases for proteins and protein–nucleic acid interactions",
  *Nucleic Acids Research* 34, 2006. doi:10.1093/nar/gkj103
- Protein sequences from the [UniProt REST API](https://rest.uniprot.org/).
- Embeddings from ESM-2 (Meta AI), via Hugging Face.
