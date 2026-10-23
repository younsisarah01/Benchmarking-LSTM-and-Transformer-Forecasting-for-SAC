# Models

The model architectures are defined in the scripts in `../scripts/`. Trained
models are written to `SAVE_DIR` (by default
`/content/drive/MyDrive/predictionsfinal/` on Google Drive).

## Implemented models

| Model | Defined in | Architecture |
|---|---|---|
| Persistence baseline | 01, 02, 03 | ŷ(t) = y(t − 1) |
| LSTM | `01_lstm_forecasting.py` | Input (60 × 5) → LSTM(128, return sequences) → Dropout(0.2) → LSTM(128) → Dropout(0.2) → Dense(64, ReLU) → Dense(1). Adam, MSE loss, up to 30 epochs, batch 256, early stopping on validation loss (patience 10). |
| Transformer encoder | `02_transformer_forecasting.py` | Input (60 × 5) → Dense(64) + sinusoidal positional encoding → 2 × [multi-head self-attention (4 heads, key dim 16, dropout 0.1) + residual + LayerNorm → FFN (128 ReLU → 64) + residual + LayerNorm] → global average pooling → Dropout(0.1) → Dense(1). Adam (lr 1e-3), MSE loss, up to 30 epochs, batch 256, early stopping (patience 10). |
| LSTM–Transformer ensemble | `03_lstm_transformer_ensemble.py` | w · LSTM + (1 − w) · Transformer, with w ∈ {0, 0.05, …, 1} selected by validation MAE. |
| SAC agent | `04_sac_energy_management.py` | 10-dimensional state. Actor: 2 × Dense(64, ReLU) → mean (tanh) and log-std heads. Twin critics: [state, action] → 2 × Dense(64, ReLU) → Dense(1), with soft-updated target networks. γ = 0.99, τ = 0.005, entropy coefficient 0.2, Adam lr 1e-4, replay buffer 50 000, batch 256. |
| Rule-based controller | `04_sac_energy_management.py` | Furnace at 60 % during 18–22 h; one 60-minute dishwasher cycle at 02:00. |

## Model files (in `SAVE_DIR`)

| File | Written by | Content |
|---|---|---|
| `scaler_lstm.pkl` | 01 | Min-Max scaler fitted on the training split, used by scripts 02, 03 and 04 |
| `lstm_checkpoint.h5` | 01 | best LSTM by validation loss |
| `model_lstm.h5` | 01 | final LSTM model |
| `model_transformer.h5` | 02 | final Transformer model |
| `ensemble_weight.txt` | 03 | selected ensemble weight w |
| `sac_seed{42,123,7}_actor.weights.h5`, `sac_seed{…}_c1.weights.h5` | 04 | best actor and critic for each seed |
| `sac_best_actor.weights.h5`, `sac_best_c1.weights.h5` | 04 | agent of the seed with the highest mean cost reduction |

Model files (`*.h5`, `*.pkl`) are excluded from version control by
`../.gitignore`. Trained weights can be shared as a release asset or through
an archive service.
