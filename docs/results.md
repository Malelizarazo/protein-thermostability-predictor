# Results

All numbers below are taken from the thesis report *Predicción de la Estabilidad Térmica de Proteínas usando
Machine Learning* (Universidad de los Andes, May 2025). Tm is in °C. Every split is 80% train / 20% validation.

## Iteration 1: ProThermDB (~11,000 entries)

A dense neural network trained on two different protein-language-model embeddings.

| Embeddings | MAE    | MSE    | RMSE   | R²      |
|------------|--------|--------|--------|---------|
| ESM-2      | 0.6610 | 0.7346 | 0.8571 | 0.3033  |
| ProtTrans  | 0.7720 | 0.9218 | 0.9601 | -0.0004 |

ESM-2 was kept for all later iterations. ProThermDB lists many mutants of the same protein as separate rows,
so the model could memorise near-identical sequences. The next iterations switched to wild-type proteins only.

## Iteration 2: Meltome Atlas subset (2,335 proteins)

Random Forest could not be trained on the full Meltome Atlas in reasonable time on CPU, so classical models
were compared on a representative subset (`data/meltome_subset_2335.csv`, script
`scripts/meltome_subset/train_random_forest.py`).

| Model          | MAE    | RMSE   | R²     | PCC    |
|----------------|--------|--------|--------|--------|
| Random Forest  | 6.1208 | 8.5679 | 0.8082 | 0.8992 |
| Bayesian Ridge | 6.5397 | 8.9346 | 0.7914 | 0.8908 |
| KNN            | 6.1149 | 8.9968 | 0.7885 | 0.8936 |
| SVR            | 6.9572 | 9.8451 | 0.7468 | 0.8689 |
| MLPRegressor   | 7.1596 | 9.7690 | 0.7507 | 0.8716 |

## Iteration 3: full Meltome Atlas wild-type set (20,000+ sequences)

Sequences were fetched from the UniProt REST API, cleaned, embedded with ESM-2 (1,280 dimensions) and used to
train neural networks in PyTorch on GPU. Code: `scripts/full_meltome/`.

| Model | MAE  | MSE   | RMSE | R²   |
|-------|------|-------|------|------|
| MLP   | 4.74 | 38.06 | 6.16 | 0.72 |
| LSTM  | 4.54 | 34.57 | 5.88 | 0.74 |

The LSTM was selected for the web app (`models/lstm_best.pt`). Both models compress predictions at the
extremes of the Tm range (for example, the LSTM saturates around 85 °C), which the report lists as the main
area for improvement.

> Iteration 2 and iteration 3 use different datasets (a 2,335-protein subset vs. the full wild-type set), so
> their metrics are not directly comparable.
