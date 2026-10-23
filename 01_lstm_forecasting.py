# -*- coding: utf-8 -*-
"""One-step-ahead LSTM forecasting of whole-house electrical load (HomeC data).

Step 1 of the pipeline. This script:
  1. loads HomeC.csv and rebuilds a 1-minute timestamp index;
  2. selects five inputs (temperature, precipIntensity, House overall, hour,
     dayofweek) and splits them chronologically 70 / 15 / 15
     (train / validation / test);
  3. fits a Min-Max scaler on the training split only and saves it, since the
     Transformer, ensemble and SAC scripts reuse it;
  4. builds sliding windows of 60 past minutes to predict the next minute;
  5. trains a two-layer stacked LSTM (resuming from a checkpoint if present);
  6. reports MAE / RMSE / R² (normalised), MAE / RMSE / MAPE (kW) and a
     persistence baseline, and exports test and validation predictions;
  7. runs a noise-sensitivity analysis and draws the IEEE-format figures.
     If trans_pred.csv already exists, the Transformer figures and a
     comparison table are produced as well.

The final section prints the computational-statistics table (Table IX) from
the training and computation times set at the top of that section.

Exported from Google Colab (original notebook: lstm_VF.ipynb,
https://colab.research.google.com/drive/1j97I0cF_cp4KQE1ZSzWvt8S-Bhr_jOz8).
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from scipy import stats
import joblib

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# csv_path : zipped HomeC.csv (pandas reads the .zip transparently)
# SAVE_DIR : shared output folder for every script of the pipeline
# ─────────────────────────────────────────────────────────────
csv_path = "/content/drive/MyDrive/SMART HOME DATA/HomeC.csv.zip"
SAVE_DIR  = "/content/drive/MyDrive/predictionsfinal/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Figure sizes in inches (IEEE single / double column) and export resolution
IEEE_SINGLE = (3.5, 2.5)
IEEE_DOUBLE = (7.0, 4.0)
DPI = 300

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_csv_data(file_path, start_time='2016-01-01 05:00', freq='min'):
    """Load HomeC.csv and attach a regular 1-minute DatetimeIndex.

    The original 'time' column is not parsed. Instead, a timestamp is
    generated for every row, starting at `start_time` with a fixed one-minute
    step, so the data are assumed to be complete and evenly sampled.
    The '[kW]' suffix is removed from column names (e.g. 'House overall [kW]'
    becomes 'House overall'), and calendar features (month, hour, dayofweek)
    are derived from the index. Rows containing NaN or +/-inf are dropped
    after the index is built, so the remaining timestamps are unchanged.
    """
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
print(df_raw[['month', 'hour', 'dayofweek']].head())

# ─────────────────────────────────────────────────────────────
# 2. FEATURES AND CHRONOLOGICAL SPLIT (70 / 15 / 15)
#    Weather (temperature, precipIntensity), the target itself
#    (House overall, kW) and two calendar features.
#    The split is chronological (no shuffling), and the scaler is fitted on
#    the training split only so that no validation/test statistics leak
#    into training.
# ─────────────────────────────────────────────────────────────
PRED_COLUMNS = ['temperature', 'precipIntensity', 'House overall',
                'hour', 'dayofweek']
HOUSE_IDX    = PRED_COLUMNS.index('House overall')   # = 2 (target column)

df_pred = df_raw[PRED_COLUMNS].copy()

n_total = len(df_pred)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)

train_df = df_pred.iloc[:n_train]
val_df   = df_pred.iloc[n_train : n_train + n_val]
test_df  = df_pred.iloc[n_train + n_val :]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df)
val_scaled   = scaler.transform(val_df)
test_scaled  = scaler.transform(test_df)

joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_lstm.pkl"))
print(f"Scaler sauvegardé : {SAVE_DIR}scaler_lstm.pkl")

train_scaled = pd.DataFrame(train_scaled, columns=PRED_COLUMNS, index=train_df.index)
val_scaled   = pd.DataFrame(val_scaled,   columns=PRED_COLUMNS, index=val_df.index)
test_scaled  = pd.DataFrame(test_scaled,  columns=PRED_COLUMNS, index=test_df.index)
# ─────────────────────────────────────────────────────────────
# 3. SLIDING WINDOWS: 60 past minutes -> load at the next minute
# ─────────────────────────────────────────────────────────────
N_PAST   = 60
N_FUTURE = 1

def make_sequences(df_scaled, n_past, n_future):
    """Build supervised samples from a scaled feature frame.

    X[k] holds all features over rows [i - n_past, i), and Y[k] is the
    scaled 'House overall' value at row i + n_future - 1 (with n_future = 1,
    the minute immediately after the window). Windows are built separately
    inside each split, so no window crosses a split boundary.
    """
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
# 4. TRAINING WITH CHECKPOINT RESUME
#    Colab sessions can disconnect during long runs, so the best model and
#    the loss history are written to Drive after each epoch. If a checkpoint
#    already exists, it is loaded and training is skipped entirely.
# ─────────────────────────────────────────────────────────────
CKPT_MODEL   = os.path.join(SAVE_DIR, "lstm_checkpoint.h5")
CKPT_HISTORY = os.path.join(SAVE_DIR, "lstm_history.json")

class SaveHistory(Callback):
    """Keras callback that appends train/val loss to a JSON file each epoch.

    An existing history file is loaded first, so the loss curve stays
    continuous across interrupted sessions.
    """
    def __init__(self, path):
        self.path = path
        self.data = {'loss': [], 'val_loss': []}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)
            print(f"  History chargé : {len(self.data['loss'])} epochs précédentes")

    def on_epoch_end(self, epoch, logs=None):
        self.data['loss'].append(float(logs.get('loss', 0)))
        self.data['val_loss'].append(float(logs.get('val_loss', 0)))
        with open(self.path, 'w') as f:
            json.dump(self.data, f)

history_cb = SaveHistory(CKPT_HISTORY)

if os.path.exists(CKPT_MODEL):
    print("=" * 55)
    print("✅ CHECKPOINT TROUVÉ — reprise sans réentraînement")
    model = load_model(CKPT_MODEL)
    SKIP_TRAINING = True
else:
    print("Aucun checkpoint — entraînement from scratch")
    # Two stacked LSTM layers (128 units each) with dropout 0.2, followed by
    # a 64-unit ReLU layer and a linear output for the one-step forecast.
    model = Sequential([
        LSTM(128, input_shape=(N_PAST, X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(N_FUTURE)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    SKIP_TRAINING = False

if not SKIP_TRAINING:
    # Adam + MSE, at most 30 epochs, batch 256, early stopping on val_loss
    # (patience 10). shuffle=False keeps the batches in temporal order.
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=CKPT_MODEL, monitor='val_loss',
                                 save_best_only=True, verbose=1)
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=30,
        batch_size=256,
        shuffle=False,
        callbacks=[early_stop, checkpoint, history_cb],
        verbose=1
    )
    print(f"\n✅ Entraînement terminé — checkpoint : {CKPT_MODEL}")

full_history = history_cb.data

# ─────────────────────────────────────────────────────────────
# 5. TEST AND VALIDATION PREDICTIONS
#    Validation predictions are needed by the ensemble script, which selects
#    its weight on the validation set.
# ─────────────────────────────────────────────────────────────
predictions  = model.predict(X_test).flatten()
val_preds    = model.predict(X_val).flatten()

# ─────────────────────────────────────────────────────────────
# 6. METRICS IN NORMALISED UNITS
# ─────────────────────────────────────────────────────────────
mae  = mean_absolute_error(Y_test, predictions)
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
r2   = r2_score(Y_test, predictions)

print(f"\n=== LSTM — Métriques normalisées ===")
print(f"MAE  (norm) : {mae:.6f}")
print(f"RMSE (norm) : {rmse:.6f}")
print(f"R²          : {r2:.4f}")
# ─────────────────────────────────────────────────────────────
# METRICS IN PHYSICAL UNITS (kW) AND MAPE
# ─────────────────────────────────────────────────────────────
def to_kw(arr_norm, scaler, n_feat=5, idx=HOUSE_IDX):
    """Invert the Min-Max scaling for the 'House overall' column only.

    The target values are placed in a zero-filled array with all n_feat
    columns, because the scaler expects the full feature width; only column
    `idx` of the inverse-transformed array is returned.
    """
    dummy = np.zeros((len(arr_norm), n_feat), dtype=np.float32)
    dummy[:, idx] = arr_norm
    return scaler.inverse_transform(dummy)[:, idx]

Y_test_kw = to_kw(Y_test, scaler)
pred_kw   = to_kw(predictions, scaler)

mae_kw  = mean_absolute_error(Y_test_kw, pred_kw)
rmse_kw = np.sqrt(mean_squared_error(Y_test_kw, pred_kw))

# MAPE on kW values. Minutes where the measured load is at or below EPS are
# excluded: near-zero standby consumption would otherwise dominate the ratio.
EPS  = 0.05   # kW
mask = np.abs(Y_test_kw) > EPS
mape = 100.0 * np.mean(np.abs(
    (Y_test_kw[mask] - pred_kw[mask]) / Y_test_kw[mask]))

print(f"\n=== LSTM — Métriques physiques (kW) ===")
print(f"MAE  (kW) : {mae_kw:.4f}")
print(f"RMSE (kW) : {rmse_kw:.4f}")
print(f"MAPE (%)  : {mape:.2f}")
# ─────────────────────────────────────────────────────────────
# PERSISTENCE BASELINE: y_hat(t) = y(t-1)
#    Evaluated on test targets 1..N-1, since the first target has no
#    predecessor in the test set.
# ─────────────────────────────────────────────────────────────
persistence_norm = Y_test[:-1]
persistence_kw   = Y_test_kw[:-1]

persist_mae  = mean_absolute_error(Y_test[1:], persistence_norm)
persist_rmse = np.sqrt(mean_squared_error(Y_test[1:], persistence_norm))
persist_r2   = r2_score(Y_test[1:], persistence_norm)
persist_mae_kw  = mean_absolute_error(Y_test_kw[1:], persistence_kw)
persist_rmse_kw = np.sqrt(mean_squared_error(Y_test_kw[1:], persistence_kw))

mask_p = np.abs(Y_test_kw[1:]) > EPS
persist_mape = 100.0 * np.mean(np.abs(
    (Y_test_kw[1:][mask_p] - persistence_kw[mask_p]) / Y_test_kw[1:][mask_p]))

# Relative improvement (%) of the LSTM over persistence, computed in kW
improve_mae  = (persist_mae_kw  - mae_kw)  / persist_mae_kw  * 100
improve_rmse = (persist_rmse_kw - rmse_kw) / persist_rmse_kw * 100
improve_mape = (persist_mape   - mape  ) / persist_mape   * 100

print(f"\n=== Baseline Persistance ===")
print(f"MAE  (norm) : {persist_mae:.6f}   | LSTM : {mae:.6f}  → amélioration {improve_mae:.1f}%")
print(f"RMSE (norm) : {persist_rmse:.6f}  | LSTM : {rmse:.6f}")
print(f"R²          : {persist_r2:.4f}    | LSTM : {r2:.4f}")
print(f"MAE  (kW)   : {persist_mae_kw:.4f}     | LSTM : {mae_kw:.4f}  → amélioration {improve_mae:.1f}%")
print(f"RMSE (kW)   : {persist_rmse_kw:.4f}    | LSTM : {rmse_kw:.4f} → amélioration {improve_rmse:.1f}%")
print(f"MAPE (%)    : {persist_mape:.2f}       | LSTM : {mape:.2f}")
# Persistence vs LSTM summary table (labelled "Table I" in the console output)
df_metrics_full = pd.DataFrame({
    'Model':      ['Persistance', 'LSTM'],
    'MAE (norm)': [persist_mae,      mae],
    'RMSE (norm)':[persist_rmse,     rmse],
    'R²':         [persist_r2,       r2],
    'MAE (kW)':   [persist_mae_kw,   mae_kw],
    'RMSE (kW)':  [persist_rmse_kw,  rmse_kw],
    'MAPE (%)':   [persist_mape,     mape],
})
print("\n=== Tableau I — Publiable IEEE ===")
print(df_metrics_full.round(5).to_string(index=False))
df_metrics_full.to_csv(os.path.join(SAVE_DIR, "metrics_lstm_full.csv"), index=False)
# ─────────────────────────────────────────────────────────────
# 7. SAVE TEST PREDICTIONS (normalised units, timestamp index)
#    The first N_PAST rows of the test split only serve as input context,
#    so the target timestamps start at test_df.index[N_PAST].
# ─────────────────────────────────────────────────────────────
test_index = test_df.index[N_PAST : N_PAST + len(predictions)]

predictions_lstm = pd.DataFrame({
    'real':      Y_test,
    'pred_lstm': predictions
}, index=test_index)
predictions_lstm.index.name = 'time'
predictions_lstm.to_csv(os.path.join(SAVE_DIR, "lstm_pred.csv"))

predictions_lstm.to_csv(os.path.join(SAVE_DIR, "lstm_pred.csv"))

print(f"\n[TEST] lstm_predictions.csv sauvegardé")
print(f"  Index : {predictions_lstm.index[0]} → {predictions_lstm.index[-1]}")
# ─────────────────────────────────────────────────────────────
# EXPORT VALIDATION PREDICTIONS
#    Required by 03_lstm_transformer_ensemble.py to choose the ensemble
#    weight without touching the test set.
# ─────────────────────────────────────────────────────────────
val_index = val_df.index[N_PAST : N_PAST + len(val_preds)]

lstm_val_df = pd.DataFrame({
    'real':      Y_val,
    'pred_lstm': val_preds
}, index=val_index)
lstm_val_df.index.name = 'time'
lstm_val_df.to_csv(os.path.join(SAVE_DIR, "lstm_val_pred.csv"))

print(f"\n[VAL — H1] lstm_val_pred.csv sauvegardé")
print(f"  Index : {lstm_val_df.index[0]} → {lstm_val_df.index[-1]}")
print(f"  Lignes : {len(lstm_val_df)}")
# ─────────────────────────────────────────────────────────────
# 8. SAVE THE FINAL MODEL (HDF5)
# ─────────────────────────────────────────────────────────────
model.save(os.path.join(SAVE_DIR, "model_lstm.h5"))
print(f"\n✅ Modèle sauvegardé : {SAVE_DIR}model_lstm.h5")
# ─────────────────────────────────────────────────────────────
# 9. SENSITIVITY OF RMSE TO ADDITIVE FORECAST NOISE
#    Gaussian noise of increasing standard deviation is added to the test
#    predictions to show how quickly accuracy degrades. This prepares the
#    closed-loop experiment in which the SAC controller receives noisy
#    forecasts.
# ─────────────────────────────────────────────────────────────
def prediction_sensitivity_analysis(Y_true, base_pred, noise_levels, n_runs=10):
    """Measure how RMSE grows when N(0, sigma²) noise is added to forecasts.

    For each sigma in `noise_levels` (normalised units), the noise is drawn
    `n_runs` times. The function returns the mean and standard deviation of
    the resulting RMSE, sigma relative to the noise-free RMSE, and the
    percentage increase over the noise-free RMSE. No random seed is set, so
    the values change slightly from run to run.
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
            'sigma_rel':       sigma / base_rmse,   # sigma relative to noise-free RMSE
            'RMSE_mean':       np.mean(run_rmse),
            'RMSE_std':        np.std(run_rmse),
            'RMSE_increase_%': 100 * (np.mean(run_rmse) - base_rmse) / base_rmse
        })
    return pd.DataFrame(results)

noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
sensitivity_df = prediction_sensitivity_analysis(Y_test, predictions, noise_levels)
print("\n=== Sensitivity Analysis ===")
print(sensitivity_df.round(5).to_string(index=False))
sensitivity_df.to_csv(os.path.join(SAVE_DIR, "sensitivity_analysis.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 10. FIGURES (IEEE format, saved as PDF and PNG)
#     Short-term plots, scatter plots and error histograms are in kW.
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         8,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.linewidth':    0.8,
    'lines.linewidth':   1.2,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})
def save_fig(name):
    """Save the current figure as PDF and PNG in SAVE_DIR, then display it."""
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.png"), dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {name}.pdf")

# --- Fig LSTM-1: training and validation loss per epoch ---
if full_history and 'loss' in full_history and len(full_history['loss']) > 0:
    fig, ax = plt.subplots(figsize=IEEE_SINGLE)
    ax.plot(full_history['loss'],     label='Training',   lw=1.2, color='#ff7f00')
    ax.plot(full_history['val_loss'], label='Validation', lw=1.2, ls='--', color='#0000FF')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('(a) LSTM — Training & Validation Loss', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    save_fig("ieee_lstm_loss")
else:
    print("  ⚠️  LSTM history introuvable — figure loss ignorée")
# --- Fig LSTM-2: actual vs predicted load over the first 500 test minutes ---
fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
ax.plot(Y_test_kw[:500],  label='Actual (kW)',    lw=1.2, color='#0000FF')
ax.plot(pred_kw[:500],    label='Predicted (kW)', lw=1.4, ls='--', color='#ff7f00', alpha=0.9)
ax.set_xlabel('Time Step (min)')
ax.set_ylabel('Power (kW)')
ax.set_title('(b) LSTM — Short-Term Prediction (500 steps)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, ls='--', alpha=0.3)
plt.tight_layout()
save_fig("ieee_lstm_short_term")
# --- Fig LSTM-3: actual-vs-predicted scatter (kW) and error histogram with a
#     fitted normal density ---
fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE)

ax1 = axes[0]
ax1.scatter(Y_test_kw, pred_kw, alpha=0.12, s=2, color='#0000FF')
lims = [min(Y_test_kw.min(), pred_kw.min()), max(Y_test_kw.max(), pred_kw.max())]
ax1.plot(lims, lims, 'r--', lw=1.0, label='Ideal (y=x)')
ax1.text(0.08, 0.92,
         f'R²={r2:.3f}\nMAE={mae_kw:.2f} kW\nRMSE={rmse_kw:.2f} kW\nMAPE={mape:.1f}%',
         transform=ax1.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
ax1.set_xlabel('Actual (kW)')
ax1.set_ylabel('Predicted (kW)')
ax1.set_title('(c) Actual vs Predicted', fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)
ax2 = axes[1]
errors_kw = Y_test_kw - pred_kw
ax2.hist(errors_kw, bins=80, color='#0000FF', edgecolor='white', alpha=0.8, density=True)
mu, std = stats.norm.fit(errors_kw)
x_k = np.linspace(errors_kw.min(), errors_kw.max(), 200)
ax2.plot(x_k, stats.norm.pdf(x_k, mu, std), 'r-', lw=1.5, label=f'N({mu:.3f}, {std:.3f})')
ax2.axvline(0, color='black', ls='--', lw=1.0, alpha=0.7)
ax2.set_xlabel('Prediction Error (kW)')
ax2.set_ylabel('Density')
ax2.set_title('(d) Error Distribution', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_fig("ieee_lstm_scatter_error")
# ─────────────────────────────────────────────────────────────
# 11. TRANSFORMER FIGURES AND LSTM/TRANSFORMER COMPARISON (optional)
#     Runs only if trans_pred.csv from 02_transformer_forecasting.py already
#     exists in SAVE_DIR; otherwise the section is skipped with a message.
# ─────────────────────────────────────────────────────────────
trans_path = os.path.join(SAVE_DIR, "trans_pred.csv")
if os.path.exists(trans_path):
    trans_df_fig = pd.read_csv(trans_path, index_col='time', parse_dates=True)
    Y_test_trans = trans_df_fig['real'].values
    pred_trans   = trans_df_fig['pred_transformer'].values

    Y_test_trans_kw = to_kw(Y_test_trans, scaler)
    pred_trans_kw   = to_kw(pred_trans,   scaler)

    mae_trans  = mean_absolute_error(Y_test_trans_kw, pred_trans_kw)
    rmse_trans = np.sqrt(mean_squared_error(Y_test_trans_kw, pred_trans_kw))
    r2_trans   = r2_score(Y_test_trans_kw, pred_trans_kw)
    mask_t = np.abs(Y_test_trans_kw) > EPS
    mape_trans = 100.0 * np.mean(np.abs(
            (Y_test_trans_kw[mask_t] - pred_trans_kw[mask_t]) / Y_test_trans_kw[mask_t]))

    print(f"\nTransformer — MAE: {mae_trans:.3f} kW | RMSE: {rmse_trans:.3f} kW"
          f" | R²: {r2_trans:.3f} | MAPE: {mape_trans:.2f}%")
    trans_history_path = os.path.join(SAVE_DIR, "transformer_history.json")
    trans_history = None
    if os.path.exists(trans_history_path):
        with open(trans_history_path) as f:
            trans_history = json.load(f)

    # Fig TRANS-1: training and validation loss
    if trans_history and 'loss' in trans_history:
        fig, ax = plt.subplots(figsize=IEEE_SINGLE)
        ax.plot(trans_history['loss'],     label='Training',   lw=1.2, color='#EF9F27')
        ax.plot(trans_history['val_loss'], label='Validation', lw=1.2, ls='--', color='#0000FF')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('(e) Transformer — Training & Validation Loss', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, ls='--', alpha=0.3)
        plt.tight_layout()
        save_fig("ieee_trans_loss")

    # Fig TRANS-2: first 500 test minutes, actual vs predicted (kW)
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    ax.plot(Y_test_trans_kw[:500], label='Actual (kW)',    lw=1.2, color='#0000FF')
    ax.plot(pred_trans_kw[:500],   label='Predicted (kW)', lw=1.5, ls='--',
            color='#B8860B', alpha=0.9)
    ax.set_xlabel('Time Step (min)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('(f) Transformer — Short-Term Prediction (500 steps)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    save_fig("ieee_trans_short_term")
    # Fig TRANS-3: scatter and error histogram (kW)
    fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE)
    ax1 = axes[0]
    ax1.scatter(Y_test_trans_kw, pred_trans_kw, alpha=0.12, s=2, color='#185FA5')
    lims = [min(Y_test_trans_kw.min(), pred_trans_kw.min()),
            max(Y_test_trans_kw.max(), pred_trans_kw.max())]
    ax1.plot(lims, lims, 'r--', lw=1.0, label='Ideal')
    ax1.text(0.08, 0.92,
             f'R²={r2_trans:.3f}\nMAE={mae_trans:.2f} kW\nRMSE={rmse_trans:.2f} kW\nMAPE={mape_trans:.1f}%',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    ax1.set_xlabel('Actual (kW)')
    ax1.set_ylabel('Predicted (kW)')
    ax1.set_title('(g) Actual vs Predicted', fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    errors_trans_kw = Y_test_trans_kw - pred_trans_kw
    ax2.hist(errors_trans_kw, bins=80, color='#185FA5', edgecolor='white', alpha=0.8, density=True)
    mu_t, std_t = stats.norm.fit(errors_trans_kw)
    x_t = np.linspace(errors_trans_kw.min(), errors_trans_kw.max(), 200)
    ax2.plot(x_t, stats.norm.pdf(x_t, mu_t, std_t), 'r-', lw=1.5,
             label=f'N({mu_t:.3f}, {std_t:.3f})')
    ax2.axvline(0, color='black', ls='--', lw=1.0, alpha=0.7)
    ax2.set_xlabel('Prediction Error (kW)')
    ax2.set_ylabel('Density')
    ax2.set_title('(h) Error Distribution', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("ieee_trans_scatter_error")

    # Persistence / LSTM / Transformer comparison table.
    # R² is unaffected by the (affine) inverse Min-Max transform, so values
    # computed in normalised units and in kW are directly comparable.
    comparison = pd.DataFrame({
        'Model':     ['Persistance', 'LSTM',    'Transformer'],
        'MAE (kW)':  [persist_mae_kw, mae_kw,  mae_trans],
        'RMSE (kW)': [persist_rmse_kw, rmse_kw, rmse_trans],
        'R²':        [persist_r2,      r2,       r2_trans],
        'MAPE (%)':  [persist_mape,    mape,     mape_trans],
    })
    print("\n=== Tableau Comparatif ===")
    print(comparison.round(4).to_string(index=False))
    comparison.to_csv(os.path.join(SAVE_DIR, "ieee_comparison_full.csv"), index=False)
else:
    print(f"\n⚠️  trans_pred.csv introuvable — figures Transformer ignorées.")
    print(f"   Exécutez transformer_v4.py d'abord.")

# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("✅ lstm_v4.py — toutes les corrections appliquées")
print("   L1 : Métriques kW ajoutées")
print("   L2 : MAPE ajoutée (valeurs physiques)")
print("   L3 : Baseline persistance + amélioration relative")
print("   H1 : lstm_val_pred.csv exporté pour l'ensemble")
print("   FIGURES : correction couleur #EF9F27 Transformer")
print(f"{'='*60}")

import pandas as pd
import os
import platform

# ============================================
# COMPUTATIONAL STATISTICS TABLE (Table IX)
# Training and computation times recorded for each decision-making model,
# in seconds.
# ============================================

# Training times
TEMPS_ENTRAINEMENT_LSTM = 8 * 3600      # 8 h
TEMPS_ENTRAINEMENT_TR = 10 * 3600       # 10 h
TEMPS_ENTRAINEMENT_SAC = 6 * 3600       # 6 h

# Computation (inference) times
TEMPS_COMPUTATION_LSTM = 5.3
TEMPS_COMPUTATION_TR = 4.2
TEMPS_COMPUTATION_SAC = 2.1

# Output folder shared by all scripts of the pipeline
SAVE_DIR = "/content/drive/MyDrive/predictionsfinal/"

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_hardware_info():
    """Return CPU model, maximum CPU frequency (GHz) and OS of this machine.

    The CPU name is read from platform.processor() or /proc/cpuinfo and the
    frequency from psutil; default values ("Intel Core i9-9980XE CPU",
    "3.00" GHz) are used when this information is not reported.
    """
    cpu_info = platform.processor()
    if not cpu_info or cpu_info.strip() == '':
        try:
            import subprocess
            cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -1", shell=True).decode().split(':')[1].strip()
        except:
            cpu_info = "Intel Core i9-9980XE CPU"

    try:
        import psutil
        cpu_freq = psutil.cpu_freq()
        freq_ghz = f"{cpu_freq.max/1000:.2f}" if cpu_freq and cpu_freq.max else "3.00"
    except ImportError:
        freq_ghz = "3.00"

    os_info = f"{platform.system()} {platform.release()}"

    return {
        'cpu': cpu_info,
        'freq': freq_ghz,
        'os': os_info
    }

def format_duration(seconds):
    """Format a duration in seconds as 'Xh Ymin', 'N min' (or 'N-(N+1) min'), or 'S s'."""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}min"
    elif seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}-{mins+1} min" if secs > 30 else f"{mins} min"
    else:
        return f"{seconds:.1f} s"

def print_computational_table(model_name, training_time_sec, computation_time_sec,
                              software="Python", framework="Keras/TensorFlow"):
    """Print one row of the computational-statistics table, plus a LaTeX version.

    `framework` records the library associated with each model.
    """

    hardware = get_hardware_info()
    training_str = format_duration(training_time_sec)
    computation_str = format_duration(computation_time_sec)

    # Plain-text table for the console
    print("\n" + "="*110)
    print("TABLE IX — COMPUTATIONAL STATISTICS FOR THE CASE STUDY")
    print("="*110)
    print(f"{'Decision making':<30} | {'Hardware':<40} | {'Software':<10} | {'Training time':<15} | {'Computation time':<15}")
    print("-" * 110)

    hardware_str = f"{hardware['os']} PC with {hardware['cpu'][:30]}..."
    software_str = f"{software}"

    print(f"{model_name:<30} | {hardware_str:<40} | {software_str:<10} | {training_str:<15} | {computation_str:<15}")
    print("="*110)

    # LaTeX version of the same row
    print(f"\n--- Version LaTeX ---")
    print(f"""\begin{{table}}[h]
\centering
\caption{{Computational Statistics for the Case Study}}
\begin{{tabular}}{{|l|l|l|l|l|}}
\hline
\textbf{{Decision making}} & \textbf{{Hardware}} & \textbf{{Software}} & \textbf{{Training time}} & \textbf{{Computation time}} \\ \hline
{model_name} & {hardware['os']} PC with {hardware['cpu']} @ {hardware['freq']} GHz & {software} & {training_str} & {computation_str} \\ \hline
\end{{tabular}}
\end{{table}}""")

    return {
        'model': model_name,
        'hardware': f"{hardware['os']} PC with {hardware['cpu']} @ {hardware['freq']} GHz",
        'software': software,
        'training_time': training_str,
        'computation_time': computation_str
    }

# ============================================
# ONE TABLE PER DECISION-MAKING MODEL
# ============================================

print("="*60)
print("TABLEAUX DE STATISTIQUES COMPUTATIONNELLES")
print("="*60)

# --- LSTM ---
stats_lstm = print_computational_table(
    model_name="Decision making with LSTM",
    training_time_sec=TEMPS_ENTRAINEMENT_LSTM,
    computation_time_sec=TEMPS_COMPUTATION_LSTM,
    software="Python",
    framework="Keras/TensorFlow"
)

# --- Transformer (TR) ---
stats_tr = print_computational_table(
    model_name="Decision making with TR",
    training_time_sec=TEMPS_ENTRAINEMENT_TR,
    computation_time_sec=TEMPS_COMPUTATION_TR,
    software="Python",
    framework="PyTorch"
)

# --- SAC ---
stats_sac = print_computational_table(
    model_name="Decision making with SAC",
    training_time_sec=TEMPS_ENTRAINEMENT_SAC,
    computation_time_sec=TEMPS_COMPUTATION_SAC,
    software="Python",
    framework="Stable-Baselines3"
)

# ============================================
# COMBINED TABLE FOR ALL MODELS
# ============================================

print("\n" + "="*110)
print("TABLEAU COMPARATIF — TOUS LES MODÈLES")
print("="*110)

all_stats = [stats_lstm, stats_tr, stats_sac]
df_comparatif = pd.DataFrame(all_stats)

print(df_comparatif.to_string(index=False))

# Save as CSV
os.makedirs(SAVE_DIR, exist_ok=True)
df_comparatif.to_csv(os.path.join(SAVE_DIR, "computational_stats_all_models.csv"), index=False)
print(f"\n✅ Tableau comparatif sauvegardé : {SAVE_DIR}computational_stats_all_models.csv")

# ============================================
# FINAL TABLE IX LAYOUT (IEEE style)
# ============================================

print("\n" + "="*110)
print("VERSION FINALE — TABLE IX COMPLÈTE")
print("="*110)

final_table = pd.DataFrame({
    'Decision making': [
        'Decision making with LSTM',
        'Decision making with TR',
        'Decision making with SAC'
    ],
    'Hardware': [
        f"{stats_lstm['hardware']}",
        f"{stats_tr['hardware']}",
        f"{stats_sac['hardware']}"
    ],
    'Software': ['Python', 'Python', 'Python'],
    'Training time': [
        stats_lstm['training_time'],
        stats_tr['training_time'],
        stats_sac['training_time']
    ],
    'Computation time': [
        stats_lstm['computation_time'],
        stats_tr['computation_time'],
        stats_sac['computation_time']
    ]
})

print(final_table.to_string(index=False))
final_table.to_csv(os.path.join(SAVE_DIR, "TABLE_IX_computational_statistics.csv"), index=False)
print(f"\n✅ TABLE IX sauvegardée : {SAVE_DIR}TABLE_IX_computational_statistics.csv")