# -*- coding: utf-8 -*-
"""One-step-ahead Transformer-encoder forecasting of whole-house load (HomeC data).

Step 2 of the pipeline. It must run after 01_lstm_forecasting.py, because it
loads the scaler fitted there (scaler_lstm.pkl). Both models therefore share
the same features, chronological 70/15/15 split, normalisation, 60-minute
input window and one-minute horizon, which is what makes their predictions
directly comparable and allows them to be combined in the ensemble.

Outputs (in SAVE_DIR): trans_pred.csv, trans_val_pred.csv,
transformer_history.json, model_transformer.h5, metrics_transformer_full.csv,
sensitivity_analysis_transformer.csv, IEEE figures, and a computational-
statistics table (Table IX) built from timings typed in by hand.

Exported from Google Colab (original notebook: Tr_VF.ipynb,
https://colab.research.google.com/drive/16bMD2s9iItGCHVHCI36N0neZG_tRO4hj).
"""

# Google Colab only: the dataset and every output file live on Google Drive.
# Outside Colab, remove these two lines and edit the paths below.
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.callbacks import EarlyStopping
from scipy import stats
import joblib

# ─────────────────────────────────────────────────────────────
# CONFIGURATION (same paths as 01_lstm_forecasting.py)
# ─────────────────────────────────────────────────────────────
csv_path = "/content/drive/MyDrive/SMART HOME DATA/HomeC.csv.zip"
SAVE_DIR  = "/content/drive/MyDrive/predictionsfinal/"
os.makedirs(SAVE_DIR, exist_ok=True)

IEEE_SINGLE = (3.5, 2.5)
IEEE_DOUBLE = (7.0, 4.0)
DPI = 300
# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING (identical to 01_lstm_forecasting.py)
# ─────────────────────────────────────────────────────────────
def load_csv_data(file_path, start_time='2016-01-01 05:00', freq='min'):
    """Load HomeC.csv with a generated 1-minute index (see 01_lstm_forecasting.py)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.replace('[kW]', '', regex=False).str.strip()
    df['time'] = pd.date_range(start=start_time, periods=len(df), freq=freq)
    df.set_index('time', inplace=True)
    df['month']     = df.index.month
    df['hour']      = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

df_raw = load_csv_data(csv_path)

PRED_COLUMNS = ['temperature', 'precipIntensity', 'House overall',
                'hour', 'dayofweek']
HOUSE_IDX    = PRED_COLUMNS.index('House overall')   # = 2 (target column)

df_pred = df_raw[PRED_COLUMNS].copy()
# ─────────────────────────────────────────────────────────────
# 2. SAME CHRONOLOGICAL 70 / 15 / 15 SPLIT AS THE LSTM
# ─────────────────────────────────────────────────────────────
n_total = len(df_pred)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)

train_df = df_pred.iloc[:n_train]
val_df   = df_pred.iloc[n_train : n_train + n_val]
test_df  = df_pred.iloc[n_train + n_val :]
# Reuse the scaler fitted on the LSTM training split (transform only, no
# refit), so both models predict in the same normalised space.
scaler = joblib.load(SAVE_DIR + "scaler_lstm.pkl")
train_scaled = scaler.transform(train_df)
val_scaled   = scaler.transform(val_df)
test_scaled  = scaler.transform(test_df)

train_scaled = pd.DataFrame(train_scaled, columns=PRED_COLUMNS, index=train_df.index)
val_scaled   = pd.DataFrame(val_scaled,   columns=PRED_COLUMNS, index=val_df.index)
test_scaled  = pd.DataFrame(test_scaled,  columns=PRED_COLUMNS, index=test_df.index)
# ─────────────────────────────────────────────────────────────
# 3. SLIDING WINDOWS — N_PAST = 60, N_FUTURE = 1, as for the LSTM
# ─────────────────────────────────────────────────────────────
N_PAST   = 60
N_FUTURE = 1

def make_sequences(df_scaled, n_past, n_future):
    """Same windowing as 01_lstm_forecasting.py: 60 past minutes -> next-minute load."""
    X, Y = [], []
    vals = df_scaled.values
    target_col = df_scaled.columns.get_loc('House overall')
    for i in range(n_past, len(vals) - n_future + 1):
        X.append(vals[i - n_past : i])
        Y.append(vals[i + n_future - 1, target_col])
    return np.array(X), np.array(Y)
X_train, Y_train = make_sequences(train_scaled, N_PAST, N_FUTURE)
X_val,   Y_val   = make_sequences(val_scaled,   N_PAST, N_FUTURE)
X_test,  Y_test  = make_sequences(test_scaled,  N_PAST, N_FUTURE)

print(f"Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")
# ─────────────────────────────────────────────────────────────
# 4. TRANSFORMER ENCODER (after Vaswani et al., 2017)
#    Linear projection to d_model, fixed sinusoidal positional encoding,
#    a stack of encoder blocks (self-attention + feed-forward, each with a
#    residual connection and post-layer normalisation), then global average
#    pooling over time and a linear output layer.
# ─────────────────────────────────────────────────────────────
def positional_encoding(length, d_model):
    """Sinusoidal positional encoding of shape (1, length, d_model).

    Even dimensions use sin, odd dimensions use cos, with wavelengths
    following 10000^(2i / d_model).
    """
    positions = np.arange(length)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)
def build_transformer(input_shape, n_future, d_model=64, num_heads=4,
                      dff=128, num_layers=2, dropout_rate=0.1):
    """Build and compile the encoder-only Transformer regressor.

    Defaults: d_model = 64, 4 attention heads (key_dim = 16), feed-forward
    width 128, 2 encoder blocks, dropout 0.1. Compiled with Adam (lr = 1e-3)
    and MSE loss.
    """
    inputs = layers.Input(shape=input_shape)

    # Project the 5 input features to d_model dimensions
    x = layers.Dense(d_model)(inputs)

    # Add the (non-trainable) positional encoding
    pos_enc = positional_encoding(input_shape[0], d_model)
    x = x + pos_enc
    # Stacked encoder blocks
    for _ in range(num_layers):
        # Multi-head self-attention, residual connection, layer norm
        attn_out = layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=d_model // num_heads,
                    dropout=dropout_rate
                )(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Position-wise feed-forward network, residual connection, layer norm
        ffn_out = layers.Dense(dff, activation='relu')(x)
        ffn_out = layers.Dropout(dropout_rate)(ffn_out)
        ffn_out = layers.Dense(d_model)(ffn_out)
        x = layers.Add()([x, ffn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Average over the 60 time steps to get one vector per window
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_future)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss='mse')
    return model

model = build_transformer((N_PAST, X_train.shape[2]), N_FUTURE)
model.summary()
# Same training budget as the LSTM: at most 30 epochs, batch 256, batches in
# temporal order, early stopping on val_loss (patience 10). Unlike the LSTM
# script, there is no checkpoint/resume logic: the model is retrained on
# every run.
early_stop = EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True)
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=256,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)
# Save the loss history; 01_lstm_forecasting.py reads it for its figures
history_dict = history.history
with open(SAVE_DIR + "transformer_history.json", 'w') as f:
    json.dump(history_dict, f)
print(f"✅ Historique sauvegardé — {len(history_dict['loss'])} epochs")

# ─────────────────────────────────────────────────────────────
# 5. TEST AND VALIDATION PREDICTIONS
#    Validation predictions are used by the ensemble script for weight
#    selection.
# ─────────────────────────────────────────────────────────────
predictions  = model.predict(X_test).flatten()
val_preds_tr = model.predict(X_val).flatten()
# ─────────────────────────────────────────────────────────────
# 6. METRICS IN NORMALISED UNITS
# ─────────────────────────────────────────────────────────────
mae  = mean_absolute_error(Y_test, predictions)
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
r2   = r2_score(Y_test, predictions)

print(f"\n=== Transformer — Métriques normalisées ===")
print(f"MAE  (norm) : {mae:.6f}")
print(f"RMSE (norm) : {rmse:.6f}")
print(f"R²          : {r2:.4f}")
# ─────────────────────────────────────────────────────────────
# METRICS IN kW AND MAPE
# ─────────────────────────────────────────────────────────────
def to_kw(arr_norm, scaler, n_feat=5, idx=HOUSE_IDX):
    """Invert the Min-Max scaling for the 'House overall' column only."""
    dummy = np.zeros((len(arr_norm), n_feat), dtype=np.float32)
    dummy[:, idx] = arr_norm
    return scaler.inverse_transform(dummy)[:, idx]

Y_test_kw = to_kw(Y_test, scaler)
pred_kw   = to_kw(predictions, scaler)

mae_kw  = mean_absolute_error(Y_test_kw, pred_kw)
rmse_kw = np.sqrt(mean_squared_error(Y_test_kw, pred_kw))

# MAPE ignores minutes with measured load <= 0.05 kW (near-zero standby)
EPS  = 0.05
mask = np.abs(Y_test_kw) > EPS
mape = 100.0 * np.mean(np.abs(
    (Y_test_kw[mask] - pred_kw[mask]) / Y_test_kw[mask]))
print(f"\n=== Transformer — Métriques physiques (kW) ===")
print(f"MAE  (kW) : {mae_kw:.4f}")
print(f"RMSE (kW) : {rmse_kw:.4f}")
print(f"MAPE (%)  : {mape:.2f}")
# ─────────────────────────────────────────────────────────────
# PERSISTENCE BASELINE: y_hat(t) = y(t-1), on test targets 1..N-1
# ─────────────────────────────────────────────────────────────
persistence_kw = Y_test_kw[:-1]
persist_mae_kw  = mean_absolute_error(Y_test_kw[1:], persistence_kw)
persist_rmse_kw = np.sqrt(mean_squared_error(Y_test_kw[1:], persistence_kw))
persist_r2      = r2_score(Y_test[1:], Y_test[:-1])
mask_p = np.abs(Y_test_kw[1:]) > EPS
persist_mape = 100.0 * np.mean(np.abs(
    (Y_test_kw[1:][mask_p] - persistence_kw[mask_p]) / Y_test_kw[1:][mask_p]))

improve_mae  = (persist_mae_kw  - mae_kw)  / persist_mae_kw  * 100
improve_rmse = (persist_rmse_kw - rmse_kw) / persist_rmse_kw * 100

print(f"\n=== Baseline Persistance ===")
print(f"MAE  (kW) : {persist_mae_kw:.4f}  | Transformer : {mae_kw:.4f}  → amélioration {improve_mae:.1f}%")
print(f"RMSE (kW) : {persist_rmse_kw:.4f} | Transformer : {rmse_kw:.4f} → amélioration {improve_rmse:.1f}%")
print(f"MAPE (%)  : {persist_mape:.2f}    | Transformer : {mape:.2f}")


df_metrics_full = pd.DataFrame({
    'Model':      ['Persistance', 'Transformer'],
    'MAE (norm)': [mean_absolute_error(Y_test[1:], Y_test[:-1]), mae],
    'RMSE (norm)':[np.sqrt(mean_squared_error(Y_test[1:], Y_test[:-1])), rmse],
    'R²':         [persist_r2,   r2],
    'MAE (kW)':   [persist_mae_kw,  mae_kw],
    'RMSE (kW)':  [persist_rmse_kw, rmse_kw],
    'MAPE (%)':   [persist_mape,    mape],
})
print("\n=== Tableau Transformer — Publiable IEEE ===")
print(df_metrics_full.round(5).to_string(index=False))
df_metrics_full.to_csv(os.path.join(SAVE_DIR, "metrics_transformer_full.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 7. SAVE TEST PREDICTIONS (normalised units, timestamp index)
# ─────────────────────────────────────────────────────────────
test_index = test_df.index[N_PAST : N_PAST + len(predictions)]

predictions_df = pd.DataFrame({
    'real':             Y_test,
    'pred_transformer': predictions
    }, index=test_index)
predictions_df.index.name = 'time'
predictions_df.to_csv(SAVE_DIR + "trans_pred.csv")

print(f"\n[TEST] trans_pred.csv sauvegardé")
print(f"  Index : {predictions_df.index[0]} → {predictions_df.index[-1]}")
# ─────────────────────────────────────────────────────────────
# EXPORT VALIDATION PREDICTIONS
#    Required by 03_lstm_transformer_ensemble.py to choose the ensemble
#    weight without using the test set.
# ─────────────────────────────────────────────────────────────
val_index = val_df.index[N_PAST : N_PAST + len(val_preds_tr)]

trans_val_df = pd.DataFrame({
    'real':             Y_val,
    'pred_transformer': val_preds_tr
    }, index=val_index)
trans_val_df.index.name = 'time'
trans_val_df.to_csv(SAVE_DIR + "trans_val_pred.csv")

print(f"\n[VAL — H1] trans_val_pred.csv sauvegardé")
print(f"  Index : {trans_val_df.index[0]} → {trans_val_df.index[-1]}")
print(f"  Lignes : {len(trans_val_df)}")
# ─────────────────────────────────────────────────────────────
# 8. SAVE THE TRAINED MODEL (HDF5)
# ─────────────────────────────────────────────────────────────
model.save(SAVE_DIR + "model_transformer.h5")
print(f"\n✅ Modèle sauvegardé : {SAVE_DIR}model_transformer.h5")

# ─────────────────────────────────────────────────────────────
# 9. FIGURES (IEEE format, PDF and PNG, values in kW)
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         8,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white'
})

def save_fig(name):
    """Save the current figure as PDF and PNG in SAVE_DIR, then display it."""
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.png"), dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {name}.pdf")

# Fig TRANS-1: training and validation loss; Figs TRANS-2 and TRANS-3 are
# drawn in the same block.
if history_dict and 'loss' in history_dict:
    fig, ax = plt.subplots(figsize=IEEE_SINGLE)
    ax.plot(history_dict['loss'],     label='Training',   lw=1.2, color='#EF9F27')
    ax.plot(history_dict['val_loss'], label='Validation', lw=1.2, ls='--', color='#0000FF')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('(e) Transformer — Training & Validation Loss', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    save_fig("ieee_trans_loss")

    # Fig TRANS-2: first 500 test minutes, actual vs predicted
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    ax.plot(Y_test_kw[:500], label='Actual (kW)',    lw=1.2, color='#0000FF')
    ax.plot(pred_kw[:500],   label='Predicted (kW)', lw=1.5, ls='--', color='#B8860B', alpha=0.9)
    ax.set_xlabel('Time Step (min)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('(f) Transformer — Short-Term Prediction (500 steps)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    save_fig("ieee_trans_short_term")

    # Fig TRANS-3: actual-vs-predicted scatter and error histogram
    fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE)

    ax1 = axes[0]
    ax1.scatter(Y_test_kw, pred_kw, alpha=0.12, s=2, color='#185FA5')
    lims = [min(Y_test_kw.min(), pred_kw.min()), max(Y_test_kw.max(), pred_kw.max())]
    ax1.plot(lims, lims, 'r--', lw=1.0, label='Ideal (y=x)')
    ax1.text(0.08, 0.92,
             f'R²={r2:.3f}\nMAE={mae_kw:.2f} kW\nRMSE={rmse_kw:.2f} kW\nMAPE={mape:.1f}%',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    ax1.set_xlabel('Actual (kW)')
    ax1.set_ylabel('Predicted (kW)')
    ax1.set_title('(g) Actual vs Predicted', fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    errors_kw = Y_test_kw - pred_kw
    ax2.hist(errors_kw, bins=80, color='#185FA5', edgecolor='white', alpha=0.8, density=True)
    mu, std = stats.norm.fit(errors_kw)
    x_k = np.linspace(errors_kw.min(), errors_kw.max(), 200)
    ax2.plot(x_k, stats.norm.pdf(x_k, mu, std), 'r-', lw=1.5, label=f'N({mu:.3f}, {std:.3f})')
    ax2.axvline(0, color='black', ls='--', lw=1.0, alpha=0.7)
    ax2.set_xlabel('Prediction Error (kW)')
    ax2.set_ylabel('Density')
    ax2.set_title('(h) Error Distribution', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("ieee_trans_scatter_error")

# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("✅ transformer_v4.py — toutes les corrections appliquées")
print(f"   L1 : Métriques kW ajoutées")
print(f"   L2 : MAPE ajoutée (valeurs physiques)")
print(f"   L3 : Baseline persistance + amélioration relative")
print(f"   H1 : trans_val_pred.csv exporté pour l'ensemble")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
# 10. SENSITIVITY OF RMSE TO ADDITIVE FORECAST NOISE — Transformer
#    Same procedure and noise levels as in 01_lstm_forecasting.py, so the
#    two models can be compared directly.
# ─────────────────────────────────────────────────────────────
def prediction_sensitivity_analysis(Y_true, base_pred, noise_levels, n_runs=10):
    """Measure how RMSE grows when N(0, sigma²) noise is added to forecasts.

    For each sigma (normalised units), noise is drawn n_runs times; the mean
    and standard deviation of the RMSE and the percentage increase over the
    noise-free RMSE are returned. Intended for the sensitivity table of the
    article (referred to as Table II in the original comments). No random
    seed is set.
    """
    base_rmse = np.sqrt(mean_squared_error(Y_true, base_pred))
    results = []
    for sigma in noise_levels:
        run_rmse = []
        for _ in range(n_runs):
            noisy = base_pred + np.random.normal(0, sigma, len(base_pred))
            run_rmse.append(np.sqrt(mean_squared_error(Y_true, noisy)))
        results.append({
            'sigma_noise':     sigma,
            'sigma_rel':       round(sigma / base_rmse, 5) if base_rmse > 0 else 0.0,
            'RMSE_mean':       np.mean(run_rmse),
            'RMSE_std':        np.std(run_rmse),
            'RMSE_increase_%': 100 * (np.mean(run_rmse) - base_rmse) / base_rmse
        })
    return pd.DataFrame(results)

noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

sensitivity_df = prediction_sensitivity_analysis(
    Y_test, predictions, noise_levels, n_runs=10
)

print("\n=== Sensitivity Analysis — Transformer ===")
print(sensitivity_df.round(5).to_string(index=False))

sensitivity_df.to_csv(
    os.path.join(SAVE_DIR, "sensitivity_analysis_transformer.csv"),
    index=False
)
print(f"\nSauvegardé : {SAVE_DIR}sensitivity_analysis_transformer.csv")

# ── LSTM vs Transformer sensitivity comparison (only if the LSTM CSV exists) ─
lstm_sens_path = os.path.join(SAVE_DIR, "sensitivity_analysis.csv")
if os.path.exists(lstm_sens_path):
    lstm_sens = pd.read_csv(lstm_sens_path)
    print("\n=== Comparaison LSTM vs Transformer — Sensitivity ===")
    compare = pd.DataFrame({
        'sigma_noise':          sensitivity_df['sigma_noise'].values,
        'RMSE_increase_LSTM_%': lstm_sens['RMSE_increase_%'].values,
        'RMSE_increase_TR_%':   sensitivity_df['RMSE_increase_%'].values,
    })
    print(compare.round(3).to_string(index=False))
    compare.to_csv(
        os.path.join(SAVE_DIR, "sensitivity_comparison_lstm_tr.csv"),
        index=False
    )
    print(f"Comparaison sauvegardée : {SAVE_DIR}sensitivity_comparison_lstm_tr.csv")

# ============================================
# TABLE IX — COMPUTATIONAL STATISTICS (Transformer only)
# Timings taken from the Transformer training logs.
# ============================================

import platform
import pandas as pd
import os

SAVE_DIR = "/content/drive/MyDrive/predictionsfinal/"

def get_hardware_info():
    """Return CPU model, max frequency (GHz) and OS.

    Default values ("Intel Xeon CPU", "2.20" GHz) are used when this
    information is not reported by the system.
    """
    cpu_info = platform.processor() or "Intel Xeon CPU"
    try:
        import psutil
        freq = psutil.cpu_freq()
        freq_ghz = f"{freq.max/1000:.2f}" if freq else "2.20"
    except:
        freq_ghz = "2.20"
    return {
        'cpu': cpu_info,
        'freq': freq_ghz,
        'os': f"{platform.system()} {platform.release()}"
    }

# Values read from the Transformer training logs
TEMPS_PAR_EPOCH_TR = 1165          # seconds per epoch (about 19.4 min on average)
NB_EPOCHS_TR = 30                  # number of training epochs
TEMPS_ENTRAINEMENT_TR = NB_EPOCHS_TR * TEMPS_PAR_EPOCH_TR  # = 34 950 s = 9 h 42 min

# Inference time = 82 s + 80 s, as recorded in the logs
TEMPS_COMPUTATION_TR = 82 + 80     # = 162 s

print(f"\n{'='*90}")
print("TABLE IX — COMPUTATIONAL STATISTICS FOR THE CASE STUDY")
print(f"{'='*90}")
print(f"{'Decision making':<30} | {'Hardware':<35} | {'Software':<10} | {'Training time':<15} | {'Computation time':<15}")
print("-" * 90)

hw = get_hardware_info()
hardware_str = f"{hw['os']} PC with {hw['cpu'][:25]}..."

# Human-readable formatting
training_h = int(TEMPS_ENTRAINEMENT_TR // 3600)
training_m = int((TEMPS_ENTRAINEMENT_TR % 3600) // 60)
training_str = f"{training_h}h {training_m}min"  # "9h 42min"

computation_str = f"{TEMPS_COMPUTATION_TR} s"  # "162 s"

print(f"{'Decision making with TR':<30} | {hardware_str:<35} | {'Python':<10} | {training_str:<15} | {computation_str:<15}")
print(f"{'='*90}")

# LaTeX version (backslashes correctly escaped here)
print(f"\n--- Version LaTeX ---")
print(f"""\\begin{{table}}[h]
\\centering
\\caption{{Computational Statistics for the Case Study}}
\\begin{{tabular}}{{|l|l|l|l|l|}}
\\hline
\\textbf{{Decision making}} & \\textbf{{Hardware}} & \\textbf{{Software}} & \\textbf{{Training time}} & \\textbf{{Computation time}} \\\\ \\hline
Decision making with TR & {hw['os']} PC with {hw['cpu']} @ {hw['freq']} GHz & Python & {training_str} & {computation_str} \\\\ \\hline
\\end{{tabular}}
\\end{{table}}""")

# Save as CSV
df_stats_tr = pd.DataFrame({
    'Decision making': ['Decision making with TR'],
    'Hardware': [f"{hw['os']} PC with {hw['cpu']} @ {hw['freq']} GHz"],
    'Software': ['Python'],
    'Training time': [training_str],
    'Computation time': [computation_str]
})
df_stats_tr.to_csv(os.path.join(SAVE_DIR, "TABLE_IX_tr_stats.csv"), index=False)
print(f"\n✅ Sauvegardé : {SAVE_DIR}TABLE_IX_tr_stats.csv")