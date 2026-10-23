# -*- coding: utf-8 -*-
"""Weighted LSTM + Transformer ensemble and comparative evaluation.

Step 3 of the pipeline. Requires the outputs of 01_lstm_forecasting.py and
02_transformer_forecasting.py in SAVE_DIR:
    scaler_lstm.pkl, lstm_pred.csv, trans_pred.csv,
    lstm_val_pred.csv, trans_val_pred.csv (validation predictions).

The ensemble is a convex combination  y_ens = w * y_LSTM + (1 - w) * y_TR.
The weight w is chosen by grid search on the validation set only and then
applied unchanged to the test set, so the test data play no part in tuning.
The script reports a persistence baseline, LSTM, Transformer and ensemble on
the same test set, both in normalised units and in kW.

Exported from Google Colab (original notebook: Hybrid_VF_.ipynb,
https://colab.research.google.com/drive/1m9b1IV-1FdOWNcUNYNKm1jw_8_CvlQas).
"""

# Google Colab only: the dataset and every output file live on Google Drive.
# Outside Colab, remove these two lines and edit the paths below.
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SAVE_DIR = "/content/drive/MyDrive/predictionsfinal/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── List what is already in SAVE_DIR and check that the validation
#    prediction files needed for weight selection are present ──────
print("="*55)
print("FICHIERS DANS SAVE_DIR :")
files = sorted(os.listdir(SAVE_DIR))
for f in files:
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    flag = "✓" if f in ["lstm_val_pred.csv","trans_val_pred.csv"] else " "
    print(f"  {flag} {f:45s} {size:>10,} o")

lstm_val_ok  = os.path.exists(SAVE_DIR + "lstm_val_pred.csv")
trans_val_ok = os.path.exists(SAVE_DIR + "trans_val_pred.csv")
print(f"\n  lstm_val_pred.csv  : {'✓ OK' if lstm_val_ok  else '✗ MANQUANT'}")
print(f"  trans_val_pred.csv : {'✓ OK' if trans_val_ok else '✗ MANQUANT — à générer'}")
print("="*55)

# ─────────────────────────────────────────────────────────────
# SCALER — fitted on the training split in 01_lstm_forecasting.py;
# used here only to map normalised predictions back to kW.
# PRED_COLUMNS must match the column order used when the scaler was fitted.
# ─────────────────────────────────────────────────────────────
scaler       = joblib.load(SAVE_DIR + "scaler_lstm.pkl")
PRED_COLUMNS = ['temperature', 'precipIntensity', 'House overall', 'hour', 'dayofweek']
HOUSE_IDX    = PRED_COLUMNS.index('House overall')

def to_kw(arr_norm):
    """Invert the Min-Max scaling for the 'House overall' column only.

    The scaler was fitted on all five input features, so the target values are
    placed in a zero-filled array of the full width before inverse_transform;
    only the target column of the result is kept.
    """
    dummy = np.zeros((len(arr_norm), len(PRED_COLUMNS)), dtype=np.float32)
    dummy[:, HOUSE_IDX] = arr_norm
    return scaler.inverse_transform(dummy)[:, HOUSE_IDX]

# ─────────────────────────────────────────────────────────────
# TEST-SET PREDICTIONS (used only for the final evaluation)
# Both models use the same split, window length (N_PAST = 60) and horizon,
# so their test predictions should be aligned row by row.
# ─────────────────────────────────────────────────────────────
lstm_test_df  = pd.read_csv(SAVE_DIR + "lstm_pred.csv")
trans_test_df = pd.read_csv(SAVE_DIR + "trans_pred.csv")

assert len(lstm_test_df) == len(trans_test_df), \
    f"Test sets de longueurs différentes : {len(lstm_test_df)} vs {len(trans_test_df)}"

# Restore the timestamp index when the CSV files carry one
if 'time' in lstm_test_df.columns:
    lstm_test_df  = lstm_test_df.set_index('time')
    trans_test_df = trans_test_df.set_index('time')

Y_test_norm  = lstm_test_df['real'].values
lstm_test    = lstm_test_df['pred_lstm'].values
trans_test   = trans_test_df['pred_transformer'].values

# ─────────────────────────────────────────────────────────────
# ENSEMBLE WEIGHT SELECTION (validation set only, to avoid test leakage)
# The validation CSVs are written by 01_lstm_forecasting.py and
# 02_transformer_forecasting.py.
# ─────────────────────────────────────────────────────────────
VAL_LSTM_PATH  = SAVE_DIR + "lstm_val_pred.csv"
VAL_TRANS_PATH = SAVE_DIR + "trans_val_pred.csv"

if os.path.exists(VAL_LSTM_PATH) and os.path.exists(VAL_TRANS_PATH):
    lstm_val_df  = pd.read_csv(VAL_LSTM_PATH)
    trans_val_df = pd.read_csv(VAL_TRANS_PATH)

    assert len(lstm_val_df) == len(trans_val_df), \
            f"Val sets de longueurs différentes : {len(lstm_val_df)} vs {len(trans_val_df)}"

    Y_val   = lstm_val_df['real'].values
    lstm_v  = lstm_val_df['pred_lstm'].values
    trans_v = trans_val_df['pred_transformer'].values

    # Grid search over w in [0, 1] with step 0.05; criterion = validation MAE
    # in normalised units.
    best_w   = 0.5
    best_mae = np.inf
    w_range  = np.arange(0.0, 1.01, 0.05)
    mae_curve_val = []

    for w in w_range:
        ens_v = w * lstm_v + (1 - w) * trans_v
        mae_v = mean_absolute_error(Y_val, ens_v)
        mae_curve_val.append(mae_v)
        if mae_v < best_mae:
            best_mae = mae_v
            best_w   = w

    print(f"[H1 CORRIGÉ] Poids optimal sélectionné sur VAL SET")
    print(f"  w_LSTM = {best_w:.2f}  |  w_Transformer = {1-best_w:.2f}")
    print(f"  MAE validation : {best_mae:.6f}")
else:
    # Without validation predictions, both models are weighted equally (w = 0.5).
    best_w    = 0.5
    w_range   = np.arange(0.0, 1.01, 0.05)
    mae_curve_val = [np.nan] * len(w_range)
    print("[AVERTISSEMENT] Fichiers val introuvables — w=0.5 utilisé par défaut.")
    print("  → Générer lstm_val_pred.csv et trans_val_pred.csv dans lstm_v3 / transformer_v3.")

# Store the selected weight so the test evaluation can be traced back to it
with open(SAVE_DIR + "ensemble_weight.txt", "w") as f:
    f.write(str(best_w))

# ─────────────────────────────────────────────────────────────
# APPLY THE FROZEN WEIGHT TO THE TEST SET
# ─────────────────────────────────────────────────────────────
ensemble_test = best_w * lstm_test + (1 - best_w) * trans_test

# ─────────────────────────────────────────────────────────────
# SAVE ENSEMBLE PREDICTIONS with the timestamp index of the LSTM file
# (normalised units, same layout as lstm_pred.csv / trans_pred.csv)
# ─────────────────────────────────────────────────────────────
out = pd.DataFrame({
    'real':          Y_test_norm,
    'pred_ensemble': ensemble_test
    }, index=lstm_test_df.index)
out.index.name = 'time'
out.to_csv(SAVE_DIR + "ensemble_pred.csv")
print(f"\n[H2 CORRIGÉ] ensemble_pred.csv sauvegardé avec index temporel")
print(f"  Colonnes : {out.columns.tolist()}")
print(f"  Index    : {out.index[0]} → {out.index[-1]}")

# ─────────────────────────────────────────────────────────────
# METRICS IN NORMALISED UNITS AND IN kW, PLUS MAPE
# ─────────────────────────────────────────────────────────────
Y_test_kw   = to_kw(Y_test_norm)
lstm_kw     = to_kw(lstm_test)
trans_kw    = to_kw(trans_test)
ensemble_kw = to_kw(ensemble_test)

def mape(y_true, y_pred, eps=0.05):
    """MAPE (%) computed on physical values (kW).

    Time steps with |y_true| <= eps kW (near-zero standby load) are excluded,
    since dividing by values close to zero would make the metric explode.
    """
    mask = np.abs(y_true) > eps
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def metrics_dict(y_true_norm, y_pred_norm, y_true_kw, y_pred_kw):
    """MAE, RMSE and R² in normalised units, MAE/RMSE in kW, and MAPE (%)."""
    return {
            'MAE (norm)':  mean_absolute_error(y_true_norm, y_pred_norm),
            'RMSE (norm)': np.sqrt(mean_squared_error(y_true_norm, y_pred_norm)),
            'R²':          r2_score(y_true_norm, y_pred_norm),
            'MAE (kW)':    mean_absolute_error(y_true_kw, y_pred_kw),
            'RMSE (kW)':   np.sqrt(mean_squared_error(y_true_kw, y_pred_kw)),
            'MAPE (%)':    mape(y_true_kw, y_pred_kw),
            }

# Persistence baseline: y_hat(t) = y(t-1). It is evaluated on test samples
# 1..N-1 because the first sample has no predecessor.
persistence_norm = Y_test_norm[:-1]
persistence_kw   = Y_test_kw[:-1]

models_metrics = {
    'Persistance': metrics_dict(
            Y_test_norm[1:], persistence_norm,
            Y_test_kw[1:],   persistence_kw),
    'LSTM':        metrics_dict(Y_test_norm, lstm_test, Y_test_kw, lstm_kw),
    'Transformer': metrics_dict(Y_test_norm, trans_test, Y_test_kw, trans_kw),
    'Ensemble':    metrics_dict(Y_test_norm, ensemble_test, Y_test_kw, ensemble_kw),
    }

df_metrics = pd.DataFrame(models_metrics).T
print("\n=== TABLE COMPARATIVE — Publiable IEEE ===")
print(df_metrics.round(5).to_string())
df_metrics.to_csv(SAVE_DIR + "metrics_4models.csv")

# ─────────────────────────────────────────────────────────────
# FIGURES (IEEE column widths in inches: single = 3.5, double = 7.16)
# ─────────────────────────────────────────────────────────────
IEEE_W1 = 3.5
IEEE_W2 = 7.16
IEEE_H  = 2.4

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         8,
    'axes.labelsize':    8,
    'axes.titlesize':    8,
    'legend.fontsize':   7,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'axes.linewidth':    0.5,
    'lines.linewidth':   1.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.28,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.5,
    'savefig.dpi':       600,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
    'figure.dpi':        150,
})

COL = {
    'Persistance': '#888888',
    'LSTM':        '#2166AC',
    'Transformer': '#1A9641',
    'Ensemble':    '#D6604D',
    'real':        '#1a1a2e',
}

def savefig(name):
    """Save the current figure as both PDF and PNG in SAVE_DIR."""
    plt.savefig(SAVE_DIR + f"{name}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(SAVE_DIR + f"{name}.png", dpi=600, bbox_inches='tight')
    plt.show()

# ── Fig 1: validation MAE as a function of the ensemble weight ───────────────
# Drawn when the validation curve has been computed.
if not any(np.isnan(v) for v in mae_curve_val):
    fig, ax = plt.subplots(figsize=(IEEE_W1, IEEE_H))
    ax.plot(w_range, mae_curve_val,
            color=COL['LSTM'], lw=1.2, marker='o', ms=3,
            label='MAE (val set)')
    ax.axvline(best_w, color=COL['Ensemble'], ls='--', lw=1.2,
               label=f'w* = {best_w:.2f}')
    ax.scatter([best_w], [best_mae], color=COL['Ensemble'], s=50, zorder=5)
    ax.set_xlabel('w (LSTM weight)')
    ax.set_ylabel('MAE (normalized)')
    ax.legend(fontsize=7)
    fig.text(0.5, 1.01,
             'Fig. 1. Ensemble weight grid search on validation set',
             ha='center', fontsize=7, style='italic')
    plt.tight_layout(pad=0.5)
    savefig("fig1_ensemble_weight_search")

# ── Fig 2: test-set error distributions (histogram + KDE) for the four models ─
errors = {
    'Persistance': Y_test_norm[1:] - persistence_norm,
    'LSTM':        Y_test_norm     - lstm_test,
    'Transformer': Y_test_norm     - trans_test,
    'Ensemble':    Y_test_norm     - ensemble_test,
}

fig, ax = plt.subplots(figsize=(IEEE_W2, IEEE_H + 0.5))
linestyles = {'Persistance': ':', 'LSTM': '--', 'Transformer': '-.', 'Ensemble': '-'}

for name, err in errors.items():
    ax.hist(err, bins=80, color=COL[name], alpha=0.30, density=True)
    kde = gaussian_kde(err, bw_method=0.25)
    x_k = np.linspace(err.min(), err.max(), 400)
    ax.plot(x_k, kde(x_k), color=COL[name], lw=1.3,
            ls=linestyles[name],
            label=f'{name}  (μ={err.mean():.4f}, σ={err.std():.4f})')

ax.axvline(0, color='black', lw=0.8, ls=':', alpha=0.6)
ax.set_xlabel('Prediction error (normalized)')
ax.set_ylabel('Density')
ax.legend(fontsize=6.5, ncol=2)
fig.text(0.5, 1.01,
         'Fig. 2. Prediction error distributions — test set',
         ha='center', fontsize=7, style='italic')
plt.tight_layout(pad=0.5)
savefig("fig2_error_distribution")

# ── Fig 3: bar chart of MAE (kW), RMSE (kW), MAPE (%) and R² per model ───────
metrics_to_plot = ['MAE (kW)', 'RMSE (kW)', 'MAPE (%)', 'R²']
model_names     = ['Persistance', 'LSTM', 'Transformer', 'Ensemble']
colors_bar      = [COL[n] for n in model_names]

fig, axes = plt.subplots(1, 4, figsize=(IEEE_W2, IEEE_H + 0.3))

for ax, metric in zip(axes, metrics_to_plot):
    vals = [df_metrics.loc[n, metric] for n in model_names]
    bars = ax.bar(range(len(model_names)), vals,
                   width=0.6, color=colors_bar, alpha=0.85,
                   edgecolor=colors_bar, linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n[:5] for n in model_names], fontsize=6)
    ax.set_title(metric, fontsize=7)
    if metric == 'R²':
        ax.set_ylim([min(vals)*0.97, 1.0])

fig.text(0.5, 1.01,
         'Fig. 3. Forecasting performance — ablation study (test set)',
         ha='center', fontsize=7, style='italic')
plt.tight_layout(pad=0.5, w_pad=0.6)
savefig("fig3_metrics_barplot")

# ── Fig 4: first 500 test steps (1 step = 1 min) in kW, one panel per model ──
t = np.arange(500)
fig, axes = plt.subplots(3, 1, figsize=(IEEE_W2, IEEE_H * 3), sharex=True)

configs = [
    ('LSTM',        lstm_kw,     COL['LSTM'],        '--'),
    ('Transformer', trans_kw,    COL['Transformer'], '-.'),
    ('Ensemble',    ensemble_kw, COL['Ensemble'],    '-'),
]

for ax, (label, pred_kw, color, ls) in zip(axes, configs):
    ax.plot(t, Y_test_kw[:500], color=COL['real'], lw=1.2, label='Real (kW)')
    ax.plot(t, pred_kw[:500],   color=color, lw=1.0, ls=ls, label=label)
    mae_v = mean_absolute_error(Y_test_kw[:500], pred_kw[:500])
    ax.set_ylabel('Power (kW)', fontsize=7)
    ax.legend(loc='upper right', fontsize=6.5, ncol=2)
    ax.text(0.02, 0.88, f'MAE = {mae_v:.3f} kW',
            transform=ax.transAxes, fontsize=6.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

axes[-1].set_xlabel('Time step (minutes)', fontsize=7)
fig.text(0.5, 1.01,
         'Fig. 4. Short-term prediction — 500 steps (physical units, kW)',
         ha='center', fontsize=7, style='italic')
plt.tight_layout(pad=0.5)
savefig("fig4_short_term_kw")

print(f"\n✅ hybrid_lstm_tr_v2 — toutes les corrections appliquées")
print(f"   w sélectionné sur : validation set (H1 corrigé)")
print(f"   w optimal         : {best_w:.2f}")
print(f"   ensemble_pred.csv : avec index temporel (H2 corrigé)")
print(f"   Métriques kW + MAPE : incluses (L1+L2 corrigés)")
print(f"   Baseline persistance : incluse (L3 corrigé)")