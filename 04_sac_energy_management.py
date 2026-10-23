# -*- coding: utf-8 -*-
"""Forecast-informed home energy management with Soft Actor-Critic (SAC).

Step 4 of the pipeline. Requires, in PRED_DIR:
    trans_pred.csv   (Transformer test predictions, 02_transformer_forecasting.py)
    scaler_lstm.pkl  (scaler fitted in 01_lstm_forecasting.py)
and the raw HomeC.csv file, which provides the appliance sub-meters.

Each episode is one day simulated at a 1-minute resolution. Measured whole-house
load is split into a non-controllable base load, the furnace (Furnace 1 +
Furnace 2) and the dishwasher. A custom SmartHomeEnv then applies:
  - a first-order thermal model of indoor temperature driven by the furnace;
  - a four-band time-of-use (TOU) tariff;
  - a reward that trades off energy cost, power above a 3.5 kW cap and
    thermal / scheduling discomfort.
The SAC agent sets the furnace duty cycle (action[0]); dishwasher cycles that
start in mid-peak or peak hours are postponed to the next off-peak period by
a fixed rule inside the environment.

Training days are HomeC days outside the Transformer test period, for which
the "forecast" seen by the agent is the measured load. Evaluation days are all
days covered by trans_pred.csv, where the agent sees the Transformer forecast.
The agent is compared with a rule-based controller and with the uncontrolled
consumption, over three random seeds.

The script was developed as a sequence of Colab cells: the training pipeline,
a figure cell that reloads the saved agent, a 48-hour dishwasher plot and a
3-seed learning curve. They are kept here in their original order.

Exported from Google Colab (original notebook: sac VF.ipynb,
https://colab.research.google.com/drive/1sJJobxkdvunmFVhhvGjDO2XWa0Ewzzul).
"""

# Google Colab only: the dataset and every output file live on Google Drive.
# Outside Colab, remove these two lines and edit the paths below.
from google.colab import drive
drive.mount('/content/drive')

import os, gc, random, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
from scipy.stats import wilcoxon
import joblib
warnings.filterwarnings('ignore')
import os, gc, random, json, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
# Let TensorFlow allocate GPU memory on demand instead of reserving it all
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# ──────────────────────────────────────────────────────────────────
# PATHS
# PRED_DIR : outputs of the forecasting scripts (predictions, scaler)
# SAVE_DIR : SAC checkpoints, result tables and figures (same folder here)
# ──────────────────────────────────────────────────────────────────
PRED_DIR = "/content/drive/MyDrive/predictionsfinal/"
SAVE_DIR = "/content/drive/MyDrive/predictionsfinal/"
HOMEC_CSV = "/content/drive/MyDrive/SMART HOME DATA/HomeC.csv.zip"
os.makedirs(SAVE_DIR, exist_ok=True)

# Only the Transformer forecasts are used by the controller.
# PRED_COL ('pred_kw') is the de-normalised forecast column that is added to
# pred_df_diag in the diagnostic block below, so that block must run first.
PRED_FILE = "trans_pred.csv"
PRED_COL = "pred_kw"

# ══════════════════════════════════════════════════════
# FORECAST LOADING AND DIAGNOSTICS
# Besides printing checks, this block builds pred_df_diag (forecasts in kW
# indexed by time), which load_day_data() uses later on.
# ══════════════════════════════════════════════════════

# 1. Load the Transformer test predictions (normalised units)
pred_df_diag = pd.read_csv(PRED_DIR + "trans_pred.csv", index_col='time', parse_dates=True)
print("Colonnes       :", pred_df_diag.columns.tolist())
print("Plage NORMALISÉE pred_transformer :",
      f"[{pred_df_diag['pred_transformer'].min():.4f}, {pred_df_diag['pred_transformer'].max():.4f}]")
print("Plage NORMALISÉE real             :",
      f"[{pred_df_diag['real'].min():.4f}, {pred_df_diag['real'].max():.4f}]")

# 2. Convert back to kW with the scaler fitted on the LSTM training split
scaler_diag = joblib.load(PRED_DIR + "scaler_lstm.pkl")
n_feat = scaler_diag.n_features_in_
house_idx = 2  # position of 'House overall' in the scaler's feature order

def denorm(arr_norm):
    """Invert the Min-Max scaling for the 'House overall' column only."""
    dummy = np.zeros((len(arr_norm), n_feat), dtype=np.float32)
    dummy[:, house_idx] = arr_norm
    return scaler_diag.inverse_transform(dummy)[:, house_idx]

real_kw  = denorm(pred_df_diag['real'].values)
pred_kw  = denorm(pred_df_diag['pred_transformer'].values)

print(f"\nAprès dénormalisation :")
print(f"  real  kW : [{real_kw.min():.3f}, {real_kw.max():.3f}] kW")
print(f"  pred  kW : [{pred_kw.min():.3f}, {pred_kw.max():.3f}] kW")

# 3. Plot the first day of the forecast period as a visual check
sample_day = str(pred_df_diag.index[0].date())
mask = pred_df_diag.index.date == pd.Timestamp(sample_day).date()
t_axis = np.arange(mask.sum())

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_axis, real_kw[mask], label='House overall — réel (kW)', color='steelblue', lw=1.5)
ax.plot(t_axis, pred_kw[mask], label='Prédiction transformer (kW)', color='orange', lw=1.2, ls='--')
ax.set_xlabel('Pas de temps (min)')
ax.set_ylabel('Puissance (kW)')
ax.set_title(f'House overall — {sample_day}')
ax.legend()
plt.tight_layout()
plt.show()

# 4. Hourly mean of measured and forecast load (kW); also stores the kW
#    columns used by the controller
pred_df_diag['real_kw'] = real_kw
pred_df_diag['pred_kw'] = pred_kw
pred_df_diag['hour'] = pred_df_diag.index.hour

hourly = pred_df_diag.groupby('hour')[['real_kw', 'pred_kw']].mean()
print("\nMoyenne horaire House overall (kW) :")
print(hourly.round(3).to_string())

# ──────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS
# ──────────────────────────────────────────────────────────────────
# Experiment: three random seeds, at most 50 training episodes per seed
# (one episode = one pass over all training days).
SEEDS = [42, 123, 7]
EPISODES = 50

# Thermal model: first-order RC model of the indoor temperature,
#   T_room(t+1) = T_room(t) + DT/TAU * ( -(T_room - T_ext) + R_THERM * P_hvac )
# with a 1-minute step expressed in hours (DT = 1/60) and TAU = R_THERM * C_THERM.
DT = 1 / 60
R_THERM = 15.0
C_THERM = 50.0
TAU = R_THERM * C_THERM
T_SET = 22.0     # comfort set-point (°C)
T_INIT = 21.0    # indoor temperature at the start of each day (°C)
P_FUR_NOM = 2.5  # nominal furnace power (kW)

# Reward weights: ALPHA for energy cost, BETA for power above P_CAP, GAMMA for
# discomfort. COST_NORM and DISCOMFORT_NORM scale each term to a comparable
# magnitude; the comfort weight and normalisation were set so that deviations
# from the set-point are penalised strongly.
ALPHA = 0.40
BETA = 0.25
GAMMA = 0.35
COST_NORM = 0.002
DISCOMFORT_NORM = 0.045

P_CAP = 3.5        # household power cap (kW) used in the demand penalty
LAMBDA_MAX = 0.30  # highest tariff (€/kWh), used to normalise prices

# Time-of-use tariff: (start hour, end hour, plot colour, label, €/kWh)
TARIF_BANDS = [
    (0, 6, '#5B9BD5', 'Off-peak', 0.08),
    (6, 12, '#70AD47', 'Shoulder', 0.12),
    (12, 18, '#FFC000', 'Mid-peak', 0.18),
    (18, 22, '#FF4444', 'Peak', 0.30),
    (22, 24, '#5B9BD5', 'Off-peak', 0.08),
]

def get_price(hour: int) -> float:
    """TOU price (€/kWh) for an hour of the day; same bands as TARIF_BANDS."""
    if 18 <= hour < 22: return 0.30
    elif 12 <= hour < 18: return 0.18
    elif 6 <= hour < 12: return 0.12
    else: return 0.08

def is_offpeak(hour: int) -> bool:
    """True for the two cheapest bands (off-peak and shoulder, <= 0.12 €/kWh)."""
    return get_price(hour) <= 0.12


# ──────────────────────────────────────────────────────────────────
# READING THE HomeC.csv SUB-METERS
# The raw file is read once and cached in memory. Timestamps are regenerated
# exactly as in the forecasting scripts (1-minute steps from 2016-01-01 05:00)
# so that dates match those of trans_pred.csv.
# ──────────────────────────────────────────────────────────────────
_homec_cache = {}

def _load_homec_raw():
    """Return the full HomeC frame (cached), or None if it cannot be read."""
    global _homec_cache
    if '__raw__' not in _homec_cache:
        try:
            raw = pd.read_csv(HOMEC_CSV, low_memory=False,
                              compression='zip' if HOMEC_CSV.endswith('.zip') else None)
            raw.columns = raw.columns.str.replace('[kW]', '', regex=False).str.strip()
            raw['time'] = pd.date_range(start='2016-01-01 05:00',
                                        periods=len(raw), freq='min')
            raw.set_index('time', inplace=True)
            _homec_cache['__raw__'] = raw
            print(f"[HomeC] Colonnes : {raw.columns.tolist()[:15]} ...")
        except Exception as e:
            print(f"  [HomeC] ERREUR : {e}")
            _homec_cache['__raw__'] = None
    return _homec_cache['__raw__']

def _get_submeter_col(raw, *candidates):
    """Return the values of the first column name found in `raw`, else None."""
    for c in candidates:
        if c in raw.columns:
            return raw[c].values
    return None

def _compute_dw_cycle_length(P_dw_real: np.ndarray, min_power: float = 0.05) -> int:
    """Median length (minutes) of the dishwasher's active periods in one day.

    A minute counts as active when the power exceeds `min_power` kW. Returns
    60 when the dishwasher does not run that day, and never less than 10.
    """
    active = (P_dw_real > min_power).astype(int)
    if active.sum() == 0:
        return 60
    changes = np.diff(active, prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) == 0:
        return 60
    return max(10, int(np.median(ends - starts)))

def load_day_data(target_day: str, noise_sigma: float = 0.0):
    """Assemble the 1-minute series needed to simulate one day.

    Returns (real_kw, pred_kw, P_base, P_fur_real, P_dw_real, T_ext_arr,
    dw_cycle_len):
      real_kw    measured whole-house load ('House overall'), clipped at 0;
      pred_kw    Transformer forecast in kW for that day. If the day is not
                 in the forecast period (training days), the measured load
                 is used instead. Optional Gaussian noise (noise_sigma, kW)
                 can be added for the sensitivity experiment;
      P_fur_real furnace load, Furnace 1 + Furnace 2;
      P_dw_real  dishwasher load;
      P_base     non-controllable remainder, real_kw - furnace - dishwasher;
      T_ext_arr  outdoor temperature (°C);
      dw_cycle_len typical dishwasher cycle length for the day (minutes).
    """
    _pred_kw_raw = None
    try:
        day_df = pred_df_diag.loc[target_day]
        if isinstance(day_df, pd.Series):
            day_df = day_df.to_frame().T
        _pred_kw_raw = day_df[PRED_COL].values.astype(np.float32)
    except KeyError:
        pass

    raw = _load_homec_raw()
    if raw is None:
        raise RuntimeError("HomeC.csv requis")

    try:
        day_raw = raw.loc[target_day]
        if isinstance(day_raw, pd.Series):
            day_raw = day_raw.to_frame().T
    except KeyError:
        raise KeyError(f"Journée {target_day} absente de HomeC.csv")

    if len(day_raw) == 0:
        raise ValueError(f"Journée {target_day} vide dans HomeC.csv")

    n = len(day_raw)

    house_col = _get_submeter_col(day_raw, 'House overall', 'house overall',
                                  'House Overall', 'use', 'Use', 'total', 'Total')
    if house_col is None:
        raise ValueError(f"Colonne 'House overall' introuvable pour {target_day}")
    real_kw = np.maximum(0.0, house_col.astype(np.float32))

    fur1 = _get_submeter_col(day_raw, 'Furnace 1', 'Furnace1', 'furnace1')
    fur2 = _get_submeter_col(day_raw, 'Furnace 2', 'Furnace2', 'furnace2')
    dw = _get_submeter_col(day_raw, 'Dishwasher', 'dishwasher', 'Dishwasher [kW]')

    fur1 = np.zeros(n, dtype=np.float32) if fur1 is None else fur1.astype(np.float32)
    fur2 = np.zeros(n, dtype=np.float32) if fur2 is None else fur2.astype(np.float32)
    dw = np.zeros(n, dtype=np.float32) if dw is None else dw.astype(np.float32)

    P_fur_real = np.maximum(0.0, fur1 + fur2)
    P_dw_real = np.maximum(0.0, dw)

    # Unit check: a median active value above 5 indicates a sub-meter in W,
    # which is then converted to kW.
    fur_active = P_fur_real[P_fur_real > 0]
    if len(fur_active) > 0 and float(np.median(fur_active)) > 5.0:
        P_fur_real = P_fur_real / 1000.0

    dw_active = P_dw_real[P_dw_real > 0]
    if len(dw_active) > 0 and float(np.median(dw_active)) > 5.0:
        P_dw_real = P_dw_real / 1000.0

    P_base = np.maximum(0.0, real_kw - P_fur_real - P_dw_real).astype(np.float32)

    def _align(arr, target_n):
        """Truncate, or pad with the last value, to exactly target_n samples."""
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) >= target_n: return arr[:target_n]
        return np.pad(arr, (0, target_n - len(arr)), mode='edge')

    if _pred_kw_raw is not None:
        pred_kw = _align(_pred_kw_raw, n)
        if noise_sigma > 0:
            pred_kw = np.clip(pred_kw + np.random.normal(0, noise_sigma, n),
                              0, None).astype(np.float32)
    else:
        pred_kw = real_kw.copy()

    # Energy-balance check: base + furnace + dishwasher equals the measured
    # total, up to the clipping of P_base at zero.
    max_err = float(np.max(np.abs(P_base + P_fur_real + P_dw_real - real_kw)))
    if max_err > 0.05:
        print(f"  [WARN] {target_day} : bilan incohérent max_err={max_err:.3f} kW")

    T_ext_col = _get_submeter_col(day_raw, 'temperature', 'Temperature', 'temp')
    T_ext_arr = T_ext_col.astype(np.float32) if T_ext_col is not None \
        else np.full(n, 10.0, dtype=np.float32)

    if T_ext_arr.mean() > 25.0:  # a daily mean above 25 is taken to mean °F
        T_ext_arr = (T_ext_arr - 32.0) * 5.0 / 9.0
        print(f"  [INFO] T_ext converti °F→°C : [{T_ext_arr.min():.1f}, {T_ext_arr.max():.1f}]°C")

    dw_cycle_len = _compute_dw_cycle_length(P_dw_real)
    return real_kw, pred_kw, P_base, P_fur_real, P_dw_real, T_ext_arr, dw_cycle_len
    print(f'  [{target_day}] pred_kw range: [{pred_kw.min():.3f}, {pred_kw.max():.3f}] kW')
# ──────────────────────────────────────────────────────────────────
# TRAINING AND EVALUATION DAYS
# ──────────────────────────────────────────────────────────────────
def get_stratified_days(df, n_train_max=15, n_eval_max=10):
    """Split the days of `df` 70/30 in time and keep at most one day per ISO week.

    Returns two lists of 'YYYY-MM-DD' strings (train, eval) of at most
    n_train_max and n_eval_max days. Taking one day per week spreads the
    selection across seasons instead of picking consecutive days.
    """
    available = sorted(df.index.normalize().unique())
    n_train_raw = max(1, int(len(available) * 0.70))
    train_raw = available[:n_train_raw]
    eval_raw = available[n_train_raw:]

    def stratify(days_list, n_max):
        weeks_seen, selected = set(), []
        for d in days_list:
            ts = pd.Timestamp(d)
            wk = (ts.isocalendar()[0], ts.isocalendar()[1])
            if wk not in weeks_seen:
                weeks_seen.add(wk)
                selected.append(str(d.date()))
                if len(selected) >= n_max:
                    break
        return selected

    tr = stratify(train_raw, n_train_max)
    ev = stratify(eval_raw, n_eval_max)
    print(f"Train : {len(tr)} jours | Eval : {len(ev)} jours")
    return tr, ev

# Evaluation days = every date covered by the Transformer test predictions.
# Training days = all other HomeC dates (i.e. the forecasting train and
# validation periods), reduced to one day per ISO week, at most 20 days.
_raw_homec = _load_homec_raw()
_pred_dates = set(pred_df_diag.index.normalize().unique())
_all_homec = set(_raw_homec.index.normalize().unique()) \
    if _raw_homec is not None else _pred_dates

_train_dates = sorted(_all_homec - _pred_dates)
_eval_dates = sorted(_pred_dates & _all_homec)

if len(_eval_dates) == 0:
    raise RuntimeError("Aucune date commune pred_df ∩ HomeC")
if len(_train_dates) == 0:
    raise RuntimeError("Aucune date de TRAIN disponible dans HomeC hors pred_df")

print(f"[INFO] TRAIN : {len(_train_dates)} jours | EVAL : {len(_eval_dates)} jours")

_train_df = pd.DataFrame(index=pd.DatetimeIndex(_train_dates))
_eval_df = pd.DataFrame(index=pd.DatetimeIndex(_eval_dates))

train_days, _ = get_stratified_days(_train_df, n_train_max=20, n_eval_max=0)
eval_days = [str(pd.Timestamp(d).date()) for d in _eval_dates]

# ──────────────────────────────────────────────────────────────────
# SCALE CHECK OF THE STATE VARIABLES (first evaluation day)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DIAGNOSTIC ÉCHELLES DES VARIABLES D'ÉTAT")
print("=" * 60)

_diag_day = eval_days[0]
real_kw, pred_kw, P_base, P_fur_real, P_dw_real, T_ext_arr, dw_cycle_len = \
    load_day_data(_diag_day)

print(f"\n📅 Journée inspectée : {_diag_day}  |  {len(real_kw)} pas de temps\n")
variables = {
    "P_base   (kW)": P_base,
    "pred_kw  (kW)": pred_kw,
    "P_fur_real(kW)": P_fur_real,
    "P_dw_real (kW)": P_dw_real,
    "real_kw  (kW)": real_kw,
    "T_ext    (°C)": T_ext_arr,
}
print(f"{'Variable':<20} {'[0]':>7} {'[1]':>7} {'[2]':>7} {'[3]':>7} {'[4]':>7} " \
      f"{'min':>7} {'max':>7} {'mean':>7} {'std':>7}")
print("-" * 90)
for name, arr in variables.items():
    vals = arr[:5]
    print(f" {name:<20} " \
          f"{vals[0]:>7.3f} {vals[1]:>7.3f} {vals[2]:>7.3f} {vals[3]:>7.3f} {vals[4]:>7.3f} " \
          f"{arr.min():>7.3f} {arr.max():>7.3f} {arr.mean():.3f} {arr.std():.3f}")

ratio_base_pred = P_base.mean() / (pred_kw.mean() + 1e-6)
print(f"\n  Ratio P_base/pred_kw = {ratio_base_pred:.1f}  " \
      f"{'⚠️  échelles différentes' if abs(ratio_base_pred - 1) > 0.5 else '✅ cohérent'}")
print(f"  P_CAP = {P_CAP} kW  |  max(P_base+P_fur) = {(P_base + P_fur_real).max():.3f} kW")
print("=" * 60 + "\n")

# ──────────────────────────────────────────────────────────────────
# REFERENCE COST AND DEMAND TRACKING
# ──────────────────────────────────────────────────────────────────
def estimate_ref_cost(real_kw: np.ndarray) -> float:
    """Daily cost (€) of the measured, uncontrolled load under the TOU tariff.

    The hour of each sample is t // 60, with t counted in minutes from the
    start of the day.
    """
    cost = 0.0
    for t, p in enumerate(real_kw):
        cost += max(0.0, float(p)) * get_price((t // 60) % 24) * DT
    return max(cost, 1e-6)

class DemandChargeTracker:
    """Rolling mean of total power over the last `window_min` minutes.

    update() returns how far that 15-minute mean exceeds P_CAP (0 otherwise).
    """
    def __init__(self, window_min=15):
        self._buf = deque(maxlen=window_min)

    def update(self, p_kw):
        self._buf.append(p_kw)
        return max(0.0, np.mean(self._buf) - P_CAP)

    def reset(self):
        self._buf.clear()

# ══════════════════════════════════════════════════════════════════
# SMART-HOME ENVIRONMENT
# ══════════════════════════════════════════════════════════════════
class SmartHomeEnv:
    """One-day home energy simulation at a 1-minute step.

    Action (2 values in [-1, 1]):
      action[0] -> furnace duty cycle in [0, 1], applied to the measured
                   furnace power, subject to comfort and peak-hour limits;
      action[1] -> second action component (the dishwasher is scheduled by
                   the rule below).
    The dishwasher is handled by a fixed rule: cycles starting (or still
    running) in mid-peak/peak hours are postponed to off-peak.

    State: see _get_state().
    """
    # Length of the state vector returned by _get_state()
    STATE_DIM = 10
    ACTION_DIM = 2
    P_DW_NOM = 1.5   # nominal dishwasher power (kW), used by the rule-based baseline

    def __init__(self, pred_kw, real_kw, P_base, P_fur_real, P_dw_real,
                 T_ext_arr, day_str="", dw_cycle_len=60):
        self.pred_kw = pred_kw.astype(np.float32)
        self.real_kw = real_kw.astype(np.float32)
        self.P_base = P_base.astype(np.float32)
        self.P_fur_real = P_fur_real.astype(np.float32)
        self.P_dw_real = P_dw_real.astype(np.float32)
        self.n_steps = len(real_kw)
        self.day_str = day_str
        self.T_ext_arr = T_ext_arr.astype(np.float32)
        self.dw_cycle_len = dw_cycle_len
        self._dc = DemandChargeTracker(15)
        self._peak_thresh = float(np.percentile(real_kw, 80))  # 80th percentile of the day's load
        self.ref_cost = estimate_ref_cost(real_kw)
        self.reset()

    def reset(self):
        self.t = 0
        self.T_room = T_INIT
        self.dw_remain = 0
        self.dw_postponed = False
        self.dw_postponed_len = 0
        self.dw_replay_t = 0
        self._in_replay = False  # True while a postponed dishwasher cycle is being replayed
        self.daily_cost = 0.0
        self._disc_dw_acc = 0.0
        self._disc_th_acc = 0.0
        self._dc.reset()

        self.hist_P_total = []
        self.hist_P_before = []
        self.hist_P_base = []
        self.hist_P_fur = []
        self.hist_P_fur_b = []
        self.hist_dc = []
        self.hist_P_dw = []
        self.hist_P_dw_b = []
        self.hist_cost_h = []
        self.hist_T_room = []
        self.hist_price = []

        self.cost_by_band = {0.08: 0.0, 0.12: 0.0, 0.18: 0.0, 0.30: 0.0}
        self.cost_ref_band = {0.08: 0.0, 0.12: 0.0, 0.18: 0.0, 0.30: 0.0}
        return self._get_state()

    def _T_ext(self, t):
        """Outdoor temperature (°C) at step t.

        Values above 50 are converted from °F. Beyond the end of the data, a
        synthetic daily sinusoid (10 ± 5 °C) is returned.
        """
        if hasattr(self, 'T_ext_arr') and t < len(self.T_ext_arr):
            val = float(self.T_ext_arr[t])
            if val > 50.0:
                val = (val - 32.0) * 5.0 / 9.0
            return val
        return 10.0 + 5.0 * np.sin(2 * np.pi * t / 1440)

    def _safe(self, arr, t, default=0.0):
        """arr[t] as float, or `default` when t is out of range."""
        return float(arr[t]) if t < len(arr) else default

    # ── State vector (10 values = STATE_DIM) ──
    def _get_state(self):
        """Observation at the current step, every term roughly in [-1, 1]:
        outdoor temperature / 40, sin and cos of the hour, price / LAMBDA_MAX,
        remaining dishwasher minutes / 60, fraction of the day elapsed,
        base load / P_CAP, forecast load / P_CAP, peak-hour flag (18-22 h),
        and headroom = (P_CAP - base - furnace) / P_CAP.
        """

        hour = (self.t // 60) % 24
        price = get_price(hour)
        T_ext = self._T_ext(self.t)
        pred = self._safe(self.pred_kw, self.t)
        Ebase = self._safe(self.P_base, self.t)
        P_fur_now = self._safe(self.P_fur_real, self.t)

        is_peak_now = 1.0 if (18 <= hour < 22) else 0.0
        headroom = P_CAP - (Ebase + P_fur_now)

        return np.array([
            T_ext / 40.0,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            price / LAMBDA_MAX,
            self.dw_remain / 60.0,
            self.t / max(self.n_steps, 1),
            Ebase / P_CAP,
            pred / P_CAP,
            is_peak_now,
            headroom / P_CAP,
        ], dtype=np.float32)

    def step(self, action):
        hour = (self.t // 60) % 24
        price = get_price(hour)
        T_ext = self._T_ext(self.t)

        # ── Furnace: map action[0] from [-1, 1] to a duty cycle in [0, 1] ──
        dc = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))

        # Comfort safeguard: if the room is more than 1 °C below the
        # set-point (below 21 °C), force a duty cycle of at least 0.8.
        if self.T_room < T_SET - 1.0:
            dc = max(dc, 0.80)

        # Tariff-dependent caps on the duty cycle, applied after the comfort
        # safeguard. `marge` is the forecast headroom below P_CAP.
        pred_now = self._safe(self.pred_kw, self.t)
        P_base_now = self._safe(self.P_base, self.t)
        marge = P_CAP - (P_base_now + pred_now)
        if 18 <= hour < 22:
            dc = min(dc, 0.60)   # peak: at most 60 %
        if 12 <= hour < 18:
            dc = min(dc, 0.80)   # mid-peak: at most 80 %

        # The duty cycle scales the furnace power measured at this minute.

        P_fur_raw = self._safe(self.P_fur_real, self.t)
        P_hvac = dc * P_fur_raw if P_fur_raw > 0.05 else 0.0
        P_hvac_b = P_fur_raw

        dT = DT / TAU * (-(self.T_room - T_ext) + R_THERM * P_hvac)
        self.T_room = float(np.clip(self.T_room + dT, 10.0, 35.0))

        # ── Dishwasher: rule-based postponement to the next off-peak period ──
        # This part does not depend on the agent's action.
        P_dw_raw  = self._safe(self.P_dw_real, self.t)
        cur_hour  = (self.t / 60.0)          # minutes since start of day -> hours
        in_peak   = (price >= 0.18)          # mid-peak (0.18) and peak (0.30)

        # Target restart time: 22:00 (start of the evening off-peak band) if it
        # is still ahead, otherwise step 0 (already in off-peak).
        if cur_hour < 22.0:
            next_offpeak_step = int(22 * 60)
        else:
            next_offpeak_step = int(0 * 60)

        P_dw = 0.0

        # Case 1: no cycle running, none pending, no demand -> nothing to do
        if P_dw_raw <= 0.05 and self.dw_remain == 0 and not self.dw_postponed:
            P_dw = 0.0

        # Case 2: a new cycle starts in the measured data
        elif P_dw_raw > 0.05 and self.dw_remain == 0 and not self.dw_postponed:
            if in_peak:
                # Expensive period: postpone the whole cycle
                self.dw_postponed      = True
                self.dw_postponed_len  = self.dw_cycle_len
                self.dw_replay_t       = self.t          # start of the measured cycle, replayed later
                self.dw_target_step    = next_offpeak_step
                P_dw = 0.0
            else:
                # Cheap period: let the cycle run as measured
                self.dw_remain  = self.dw_cycle_len
                self._in_replay = False
                P_dw            = P_dw_raw
                self.dw_remain -= 1

        # Case 3: a cycle is running
        elif self.dw_remain > 0:
            if in_peak:
                # Price rose to mid-peak/peak mid-cycle: pause and postpone the rest
                self.dw_postponed      = True
                self.dw_postponed_len  = self.dw_remain
                self.dw_target_step    = next_offpeak_step
                self.dw_remain         = 0
                P_dw = 0.0
            else:
                if self._in_replay:
                    P_dw = self._safe(self.P_dw_real, self.dw_replay_t)
                    self.dw_replay_t += 1
                else:
                    P_dw = P_dw_raw if P_dw_raw > 0.01 else 0.0
                self.dw_remain -= 1

        # Case 4: a cycle is postponed -> wait for off-peak, or force it to run
        # if only just enough minutes remain to finish it before the day ends
        elif self.dw_postponed:
            steps_left = self.n_steps - self.t
            must_run   = (steps_left <= self.dw_postponed_len + 5)
            at_target  = (self.t >= self.dw_target_step)

            if (at_target and not in_peak) or must_run:
                # Replay the postponed cycle from its recorded start
                self.dw_remain    = self.dw_postponed_len
                self.dw_postponed = False
                self._in_replay   = True
                P_dw = self._safe(self.P_dw_real, self.dw_replay_t)
                self.dw_replay_t += 1
                self.dw_remain   -= 1
            else:
                P_dw = 0.0

        # ── Power balance and cost (controlled vs measured reference) ──
        P_base_t = self._safe(self.P_base, self.t)
        P_total = max(0.0, P_base_t + P_hvac + P_dw)
        P_ref_t = max(0.0, float(self._safe(self.P_base, self.t)) \
                      + float(self._safe(self.P_fur_real, self.t)) \
                      + float(self._safe(self.P_dw_real, self.t)))

        cost_stp = P_total * price * DT
        cost_ref = P_ref_t * price * DT

        d_exc = self._dc.update(P_total)

        # Discomfort: rho_dw penalises a dishwasher cycle that is still waiting
        # during mid-peak/peak hours; rho_f is the relative deviation of the
        # indoor temperature from the set-point.
        rho_dw = (price / LAMBDA_MAX) if (self.dw_postponed and price >= 0.18) else 0.0
        rho_f = abs(self.T_room - T_SET) / T_SET
        rho_t = rho_dw + rho_f

        # ── Reward: weighted sum of normalised cost, cap excess and discomfort ──
        # The demand term uses the instantaneous excess over P_CAP; d_exc is
        # the 15-minute rolling excess.
        term_cost = cost_stp / COST_NORM
        excess = max(0.0, P_total - P_CAP)
        term_demand = (excess / P_CAP) * 25.0
        term_comfort = rho_t / DISCOMFORT_NORM
        reward = -(ALPHA * term_cost + BETA * term_demand + GAMMA * term_comfort)

        self.daily_cost += cost_stp
        # Daily discomfort accumulators (DT * 60 = 1 minute per step)
        self._disc_dw_acc += rho_dw * DT * 60
        self._disc_th_acc += rho_f * DT * 60
        self.cost_by_band[price] += cost_stp
        self.cost_ref_band[price] += cost_ref

        self.hist_P_total.append(P_total)
        self.hist_P_before.append(P_ref_t)
        self.hist_P_base.append(P_base_t)
        self.hist_P_fur.append(P_hvac)
        self.hist_P_fur_b.append(P_hvac_b)
        self.hist_dc.append(dc)
        self.hist_P_dw.append(P_dw)
        self.hist_cost_h.append(cost_stp)
        self.hist_T_room.append(self.T_room)
        self.hist_price.append(price)
        self.hist_P_dw_b.append(float(self.P_dw_real[self.t] if self.t < len(self.P_dw_real) else 0.0))

        self.t += 1
        done = (self.t >= self.n_steps)
        return self._get_state(), reward, done, {}

# ══════════════════════════════════════════════════════════════════
# SAC NETWORKS AND AGENT
# ══════════════════════════════════════════════════════════════════
def build_actor(sd, ad):
    """Gaussian policy network: 2 x 64 ReLU layers.

    Outputs [mu, log_std] concatenated. mu is squashed by tanh; log_std is
    linear and clipped later to [LOG_STD_MIN, LOG_STD_MAX].
    """
    inp = layers.Input(shape=(sd,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    mu = layers.Dense(ad, activation='tanh')(x)
    log_s = layers.Dense(ad, activation="linear", name="log_std")(x)
    out = layers.Concatenate(name="actor_out")([mu, log_s])
    return tf.keras.Model(inp, out)

def build_critic(sd, ad):
    """Q-network Q(s, a): state and action concatenated, 2 x 64 ReLU layers."""
    s = layers.Input(shape=(sd,))
    a = layers.Input(shape=(ad,))
    x = layers.Concatenate()([s, a])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1)(x)
    return tf.keras.Model([s, a], out)

class ReplayBuffer:
    """FIFO experience replay with uniform random sampling."""
    def __init__(self, capacity=50_000):
        self._buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self._buf.append((s.astype(np.float32), a.astype(np.float32),
                          np.float32(r), s_.astype(np.float32), np.float32(done)))

    def sample(self, bs):
        batch = random.sample(self._buf, bs)
        s, a, r, s_, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.float32),
                np.array(r, dtype=np.float32)[:, None],
                np.array(s_, dtype=np.float32),
                np.array(d, dtype=np.float32)[:, None])

    def __len__(self):
        return len(self._buf)

class SACAgent:
    """Soft Actor-Critic with twin critics and soft-updated target critics.

    Hyper-parameters: discount gamma = 0.99, target update rate tau = 0.005,
    fixed entropy coefficient alpha_ent = 0.2 (no automatic tuning),
    Adam with lr = 1e-4, replay buffer of 50 000 transitions, batch size 256.

    Implementation notes:
      - actions are sampled as clip(mu + std * eps, -1, 1);
      - log pi is approximated by the Gaussian quadratic term
        -0.5 * sum(((a - mu) / std)^2), without the normalising constant or a
        tanh change-of-variables correction;
      - save()/load() only store the actor and the first critic.
    """
    LOG_STD_MIN = -4.0
    LOG_STD_MAX = 0.5

    def __init__(self, sd, ad, gamma=0.99, tau=0.005, alpha_ent=0.2, lr=1e-4):
        self.gamma, self.tau, self.alpha_ent = gamma, tau, alpha_ent
        self.ad = ad
        self.actor = build_actor(sd, ad)
        self.critic1 = build_critic(sd, ad)
        self.critic2 = build_critic(sd, ad)
        self.critic1_target = build_critic(sd, ad)
        self.critic2_target = build_critic(sd, ad)
        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())
        self.opt_a = tf.keras.optimizers.Adam(lr)
        self.opt_c1 = tf.keras.optimizers.Adam(lr)
        self.opt_c2 = tf.keras.optimizers.Adam(lr)
        self.buffer = ReplayBuffer(50_000)
        self.batch_size = 256

    @tf.function(reduce_retracing=True)
    def _sample_action_tf(self, s):
        """Stochastic action used for exploration during training."""
        _out = self.actor(s)
        mu, ls = _out[:, :self.ad], _out[:, self.ad:]
        std = tf.exp(tf.clip_by_value(ls, self.LOG_STD_MIN, self.LOG_STD_MAX))
        return tf.clip_by_value(mu + std * tf.random.normal(mu.shape), -1, 1)

    def select_action(self, state, deterministic=False):
        """Return an action for one state; deterministic=True uses the mean (evaluation)."""
        s = tf.constant(state[None], dtype=tf.float32)
        _out = self.actor(s)
        mu, ls = _out[:, :self.ad], _out[:, self.ad:]
        if deterministic:
            return np.clip(mu.numpy()[0], -1, 1).astype(np.float32)
        return self._sample_action_tf(s).numpy()[0].astype(np.float32)

    @tf.function
    def _update_critics(self, s, a, r, s_, done):
        """Regress both critics on the soft Bellman target, using the minimum
        of the two target critics and the entropy bonus."""
        _out_ = self.actor(s_)
        mu_, ls_ = _out_[:, :self.ad], _out_[:, self.ad:]
        std_ = tf.exp(tf.clip_by_value(ls_, self.LOG_STD_MIN, self.LOG_STD_MAX))
        a_ = tf.clip_by_value(mu_ + std_ * tf.random.normal(mu_.shape), -1, 1)
        log_pi_ = -0.5 * tf.reduce_sum(
            tf.square((a_ - mu_) / (std_ + 1e-6)), 1, keepdims=True)
        y = r + self.gamma * (1 - done) * (
            tf.minimum(self.critic1_target([s_, a_]),
                       self.critic2_target([s_, a_])) - self.alpha_ent * log_pi_)
        for crit, opt in [(self.critic1, self.opt_c1), (self.critic2, self.opt_c2)]:
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(crit([s, a]) - y))
                opt.apply_gradients(zip(
                    tape.gradient(loss, crit.trainable_variables),
                    crit.trainable_variables))

    @tf.function
    def _update_actor(self, s):
        """Minimise alpha * log pi - min(Q1, Q2) for actions sampled from the policy."""
        with tf.GradientTape() as tape:
            _out = self.actor(s)
            mu, ls = _out[:, :self.ad], _out[:, self.ad:]
            std = tf.exp(tf.clip_by_value(ls, self.LOG_STD_MIN, self.LOG_STD_MAX))
            a_samp = tf.clip_by_value(mu + std * tf.random.normal(mu.shape), -1, 1)
            log_pi = -0.5 * tf.reduce_sum(
                tf.square((a_samp - mu) / (std + 1e-6)), 1, keepdims=True)
            loss_a = tf.reduce_mean(
                self.alpha_ent * log_pi
                - tf.minimum(self.critic1([s, a_samp]),
                             self.critic2([s, a_samp])))
            self.opt_a.apply_gradients(zip(
                tape.gradient(loss_a, self.actor.trainable_variables),
                self.actor.trainable_variables))

    def update(self):
        """One gradient step for the critics and the actor, then Polyak-average the targets.

        The critics and the actor are updated on two different random batches.
        """
        if len(self.buffer) < self.batch_size:
            return
        self._update_critics(*self.buffer.sample(self.batch_size))
        self._update_actor(self.buffer.sample(self.batch_size)[0])
        for tgt, src in [(self.critic1_target, self.critic1),
                         (self.critic2_target, self.critic2)]:
            for tw, sw in zip(tgt.trainable_variables, src.trainable_variables):
                tw.assign(self.tau * sw + (1 - self.tau) * tw)

    def save(self, path):
        self.actor.save_weights(path + "_actor.weights.h5")
        self.critic1.save_weights(path + "_c1.weights.h5")

    def load(self, path):
        self.actor.load_weights(path + "_actor.weights.h5")
        self.critic1.load_weights(path + "_c1.weights.h5")

# ──────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ──────────────────────────────────────────────────────────────────
def make_env(day_str: str, noise_sigma: float = 0.0) -> SmartHomeEnv:
    """Build a SmartHomeEnv for one day (optionally with noisy forecasts)."""
    real_kw, pred_kw, P_base, P_fur_real, P_dw_real, T_ext_arr, dw_cycle_len = \
        load_day_data(day_str, noise_sigma)
    return SmartHomeEnv(pred_kw, real_kw, P_base, P_fur_real, P_dw_real,
                        T_ext_arr, day_str, dw_cycle_len)

# ──────────────────────────────────────────────────────────────────
# TRAINING (best-agent checkpointing, early stopping, resume)
# ──────────────────────────────────────────────────────────────────
def train_sac_multijour(agent, days, episodes=EPISODES, update_freq=4,
                        verbose=True, save_path=None, patience=10):
    """Train a SAC agent over several days per episode.

    One episode visits every training day once, in random order; the agent is
    updated every `update_freq` environment steps. Returns the list of episode
    rewards and the index of the best episode.

    - If a checkpoint exists at save_path, the weights and reward history are
      reloaded and training continues from the next episode.
    - Every second episode, the best agent so far (highest episode reward)
      is saved, early stopping is checked (`patience` checks without
      improvement), and the learning rate is lowered once to 5e-5 if the
      rolling reward falls below 85 % of the mean of the first four episodes.
    """
    rewards_hist = []
    best_reward = -np.inf
    no_improve = 0
    start_episode = 0

    _ckpt_weights = (save_path + "_actor.weights.h5") if save_path else None
    _ckpt_history = (save_path + "_rewards_hist.json") if save_path else None

    # ── Resume from an existing checkpoint ──
    if save_path and os.path.exists(_ckpt_weights) and os.path.exists(_ckpt_history):
        try:
            agent.load(save_path)
            with open(_ckpt_history) as _f:
                _hist_data = json.load(_f)
                rewards_hist = _hist_data.get('rewards', [])
                start_episode = len(rewards_hist)
                best_reward = max(rewards_hist) if rewards_hist else -np.inf
                print(f"  [REPRISE] Checkpoint chargé → épisode {start_episode}, " \
                      f"meilleur reward = {best_reward:.3f}")
        except Exception as _e:
            print(f"  [REPRISE] Échec chargement : {_e} — entraînement from scratch")
            rewards_hist = []
            start_episode = 0
            best_reward = -np.inf
    else:
        print(f"  [TRAIN] Pas de checkpoint — entraînement from scratch")

    day_strs = [str(d)[:10] for d in days]
    best_ep = start_episode
    _lr_reduced = False

    for ep in range(start_episode, episodes):
        ep_reward = 0.0
        random.shuffle(day_strs)

        for day_str in day_strs:
            try:
                real_kw, pred_kw, P_base, P_fur_real, P_dw_real, T_ext_arr, dw_cycle_len = load_day_data(day_str)
            except (KeyError, ValueError, RuntimeError):
                continue

            env = SmartHomeEnv(pred_kw, real_kw, P_base, P_fur_real, P_dw_real, T_ext_arr, day_str, dw_cycle_len)
            state = env.reset()
            while True:
                action = agent.select_action(state)
                ns, reward, done, _ = env.step(action)
                agent.buffer.push(state, action, reward, ns, float(done))
                if env.t % update_freq == 0:
                    agent.update()
                ep_reward += reward
                state = ns
                if done:
                    break
            del env
            gc.collect()

        rewards_hist.append(ep_reward)

        w = max(4, min(8, len(rewards_hist) // 3))
        roll = float(np.mean(rewards_hist[-w:]))

        if verbose and (ep - start_episode + 1) % 2 == 0:
            print(f"  Ep {ep+1:3d} | R={ep_reward:.3f} | Roll({w})={roll:.3f} " \
                  f"| Best={best_reward:.3f} | No-impr={no_improve}/{patience}")

            # ── Save the best agent so far ──
            if ep_reward > best_reward and save_path:
                best_reward = ep_reward
                best_ep = ep
                no_improve = 0
                agent.save(save_path)
                with open(_ckpt_history, 'w') as _f:
                    json.dump({'rewards': rewards_hist, 'best_ep': best_ep}, _f)
                if verbose:
                    print(f"    ✅ Meilleur agent sauvegardé (ep {ep+1}, R={best_reward:.3f})")
            else:
                no_improve += 1

            # ── Early stopping ──
            if no_improve >= patience:
                print(f"\n  ⏹ Early stopping : {patience} épisodes sans amélioration.")
                print(f"     Meilleur épisode : {best_ep+1} (R={best_reward:.3f})")
                print(f"     Meilleur agent conservé dans : {save_path}")
                print(f"     Relancer cette fonction reprendra automatiquement depuis ce point.")
                break

            # ── Lower the learning rate once if rewards deteriorate ──
            if not _lr_reduced and len(rewards_hist) >= 8:
                roll_early = float(np.mean(rewards_hist[:4]))
                if roll < roll_early * 0.85:
                    new_lr = 5e-5
                    for opt in [agent.opt_a, agent.opt_c1, agent.opt_c2]:
                        opt.learning_rate.assign(new_lr)
                    _lr_reduced = True
                    if verbose:
                        print(f"    📉 lr réduit → {new_lr} (divergence détectée)")

    return rewards_hist, best_ep

# ══════════════════════════════════════════════════════════════════
# SAC EVALUATION
# evaluate_multijour, evaluate_rule_based and run_experiment are defined a
# second time further below (Colab cell order); the later definitions are
# the ones called by the main pipeline.
# ══════════════════════════════════════════════════════════════════
def evaluate_multijour(agent, days, condition='SAC-transformer',
                       noise_sigma=0.0, keep_last_env=False):
    results, last_env_data = [], None
    for day_str in [str(d)[:10] for d in days]:
        try:
            env = make_env(day_str, noise_sigma)
        except KeyError:
            continue
        state = env.reset()
        while True:
            action = agent.select_action(state, deterministic=True)
            state, _, done, _ = env.step(action)
            if done:
                break

        cost_red = (env.ref_cost - env.daily_cost) / env.ref_cost * 100.0
        results.append({
            'day': day_str,
            'condition': condition,
            'daily_cost_eur': float(env.daily_cost),
            'ref_cost_eur': float(env.ref_cost),
            'cost_reduction_%': float(cost_red),
            'peak_kw': float(np.max(env.hist_P_total)),
            'peak_before_kw': float(np.max(env.hist_P_before)),
            'discomfort_th': float(env._disc_th_acc),
            'discomfort_dw': float(env._disc_dw_acc),
            'cost_offpeak': float(env.cost_by_band[0.08]),
            'cost_shoulder': float(env.cost_by_band[0.12]),
            'cost_midpeak': float(env.cost_by_band[0.18]),
            'cost_peak': float(env.cost_by_band[0.30]),
            'ref_offpeak': float(env.cost_ref_band[0.08]),
            'ref_shoulder': float(env.cost_ref_band[0.12]),
            'ref_midpeak': float(env.cost_ref_band[0.18]),
            'ref_peak': float(env.cost_ref_band[0.30]),
        })
        if keep_last_env:
            last_env_data = {
                'P_total':      np.array(env.hist_P_total),
                'P_before':     np.array(env.hist_P_before),
                'P_base':       np.array(env.hist_P_base),
                'P_fur':        np.array(env.hist_P_fur),
                'P_fur_b':      np.array(env.hist_P_fur_b),
                'dc':           np.array(env.hist_dc),
                'P_dw':         np.array(env.hist_P_dw),
                'P_dw_b':       env.P_dw_real[:len(env.hist_P_dw)].copy(),  # measured dishwasher load
                'cost_h':       np.array(env.hist_cost_h),
                'T_room':       np.array(env.hist_T_room),
                'price':        np.array(env.hist_price),
                'day_str':      day_str,
                'cost_by_band': dict(env.cost_by_band),
                'cost_ref_band':dict(env.cost_ref_band),
                'ref_cost':     float(env.ref_cost),
                'mean_kw':      float(np.mean(env.hist_P_total)),
                'mean_before_kw': float(np.mean(env.hist_P_before)),
                'pred_kw':      env.pred_kw.tolist(),
            }
    del env
    gc.collect()
    return pd.DataFrame(results), last_env_data

# ──────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE
# ──────────────────────────────────────────────────────────────────
def evaluate_rule_based(days):
    """See the second definition of evaluate_rule_based below."""
    results = []
    for day_str in [str(d)[:10] for d in days]:
        try:
            real_kw, _, P_base, P_fur_real, P_dw_real, T_ext_arr, _ = load_day_data(day_str)
        except KeyError:
            continue
        ref_cost = estimate_ref_cost(real_kw)
        total_cost = 0.0
        discomfort = 0.0
        T_room = T_INIT
        peaks = []
        dw_remain = 0
        dc_obj = DemandChargeTracker(15)
        cost_bands = {0.08: 0.0, 0.12: 0.0, 0.18: 0.0, 0.30: 0.0}
        ref_bands = {0.08: 0.0, 0.12: 0.0, 0.18: 0.0, 0.30: 0.0}

        for t in range(len(real_kw)):
            hour = (t // 60) % 24
            price = get_price(hour)
            dc_r = 0.6 if (18 <= hour < 22) else 1.0
            P_hvac = dc_r * float(P_fur_real[t])
            T_ext = float(T_ext_arr[t]) if t < len(T_ext_arr) else 10.0
            dT = DT / TAU * (-(T_room - T_ext) + R_THERM * P_hvac)
            T_room = float(np.clip(T_room + dT, 10.0, 35.0))

            if t == 120:
                dw_remain = 60

            P_dw_raw = float(P_dw_real[t])
            P_dw_active = P_dw_raw if P_dw_raw > 0.05 else SmartHomeEnv.P_DW_NOM
            P_dw = P_dw_active if dw_remain > 0 else 0.0
            if dw_remain > 0:
                dw_remain -= 1

            P_total = max(0.0, float(P_base[t]) + P_hvac + P_dw)
            cost_t = P_total * price * DT

            total_cost += cost_t
            peaks.append(P_total)
            dc_obj.update(P_total)
            discomfort += (abs(T_room - T_SET) / T_SET) * DT * 60
            cost_bands[price] += cost_t
            ref_bands[price] += float(real_kw[t]) * price * DT

        results.append({
            'day': day_str,
            'condition': 'Rule-Based',
            'daily_cost_eur': total_cost,
            'ref_cost_eur': ref_cost,
            'cost_reduction_%': (ref_cost - total_cost) / ref_cost * 100.0,
            'peak_kw': float(np.max(peaks)),
            'peak_before_kw': float(np.max(real_kw)),
            'discomfort_th': discomfort,
            'discomfort_dw': 0.0,
            'cost_offpeak': cost_bands[0.08],
            'cost_shoulder': cost_bands[0.12],
            'cost_midpeak': cost_bands[0.18],
            'cost_peak': cost_bands[0.30],
            'ref_offpeak': ref_bands[0.08],
            'ref_shoulder': ref_bands[0.12],
            'ref_midpeak': ref_bands[0.18],
            'ref_peak': ref_bands[0.30],
        })
    return pd.DataFrame(results)

# ──────────────────────────────────────────────────────────────────
# MULTI-SEED EXPERIMENT
# ──────────────────────────────────────────────────────────────────
def run_experiment(train_days, eval_days, seeds=SEEDS, episodes=EPISODES):
    """See the second definition of run_experiment below."""
    all_results = []
    best_agent = None
    best_eval = -np.inf
    all_rewards = []

    for seed in seeds:
        print(f"\n{'='*50}\n  SEED {seed}\n{'='*50}")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        ckpt_path = SAVE_DIR + f"sac_seed{seed}"
        agent = SACAgent(SmartHomeEnv.STATE_DIM, SmartHomeEnv.ACTION_DIM)

        rewards_hist, best_ep = train_sac_multijour(
            agent, train_days,
            episodes=episodes,
            update_freq=4,
            save_path=ckpt_path,
            patience=10,
            verbose=True,
        )
        all_rewards.append(rewards_hist)

        # Reload the best checkpoint of this seed (not the last episode)
        if os.path.exists(ckpt_path + "_actor.weights.h5"):
            agent.load(ckpt_path)
            print(f"  [EVAL] Meilleur agent seed {seed} rechargé (épisode {best_ep+1})")

        kpis_sac, _ = evaluate_multijour(agent, eval_days, 'SAC-transformer')
        kpis_rules = evaluate_rule_based(eval_days)

        for df_kpi, cond in [(kpis_sac, 'SAC-transformer'), (kpis_rules, 'Rule-Based')]:
            df_kpi = df_kpi.copy()
            df_kpi['seed'] = seed
            df_kpi['condition'] = cond
            all_results.append(df_kpi)

        eval_mean = float(kpis_sac['cost_reduction_%'].mean())
        print(f"  [SEED {seed}] cost_reduction moyen = {eval_mean:.2f}%")

        if eval_mean > best_eval:
            best_eval = eval_mean
            best_agent = agent
            agent.save(SAVE_DIR + "sac_best")
            print(f"  ✅ Nouveau meilleur global (seed {seed}, réduction {best_eval:.2f}%")

        gc.collect()

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(SAVE_DIR + "results_seeds.csv", index=False)

    best_seed_idx = int(np.argmax([
        np.mean(r[-5:]) if len(r) >= 5 else np.mean(r) for r in all_rewards
    ]))
    return df_all, best_agent, all_rewards[best_seed_idx]

# ──────────────────────────────────────────────────────────────────
# SAC EVALUATION (used by the main pipeline)
# ──────────────────────────────────────────────────────────────────
def evaluate_multijour(agent, days, condition='SAC-transformer',
                       noise_sigma=0.0, keep_last_env=False):
    """Run the agent deterministically on each day and collect daily KPIs.

    Per day: cost with control and reference cost of the measured load (€),
    cost reduction (%), peak power after and before control (kW), thermal and
    dishwasher discomfort, and cost per tariff band (controlled and reference).
    With keep_last_env=True, the full minute-by-minute trajectories of the
    last simulated day are also returned (used for the figures).
    """
    results, last_env_data = [], None
    for day_str in [str(d)[:10] for d in days]:
        try:
            env = make_env(day_str, noise_sigma)
        except KeyError:
            continue
        state = env.reset()
        while True:
            action      = agent.select_action(state, deterministic=True)
            state, _, done, _ = env.step(action)
            if done:
                break

        cost_red = (env.ref_cost - env.daily_cost) / env.ref_cost * 100.0
        results.append({
            'day':              day_str,
            'condition':        condition,
            'daily_cost_eur':   float(env.daily_cost),
            'ref_cost_eur':     float(env.ref_cost),
            'cost_reduction_%': float(cost_red),
            'peak_kw':          float(np.max(env.hist_P_total)),
            'peak_before_kw':   float(np.max(env.hist_P_before)),
            'discomfort_th':    float(env._disc_th_acc),
            'discomfort_dw':    float(env._disc_dw_acc),
            'cost_offpeak':     float(env.cost_by_band[0.08]),
            'cost_shoulder':    float(env.cost_by_band[0.12]),
            'cost_midpeak':     float(env.cost_by_band[0.18]),
            'cost_peak':        float(env.cost_by_band[0.30]),
            'ref_offpeak':      float(env.cost_ref_band[0.08]),
            'ref_shoulder':     float(env.cost_ref_band[0.12]),
            'ref_midpeak':      float(env.cost_ref_band[0.18]),
            'ref_peak':         float(env.cost_ref_band[0.30]),
        })
    if keep_last_env:
        last_env_data = {
            'P_total':       np.array(env.hist_P_total),
            'P_before':      np.array(env.hist_P_before),
            'P_base':        np.array(env.hist_P_base),
            'P_fur':         np.array(env.hist_P_fur),
            'P_fur_b':       np.array(env.hist_P_fur_b),
            'dc':            np.array(env.hist_dc),
            'P_dw':          np.array(env.hist_P_dw),
            'P_dw_b':        np.array(env.hist_P_dw_b),
            'cost_h':        np.array(env.hist_cost_h),
            'T_room':        np.array(env.hist_T_room),
            'price':         np.array(env.hist_price),
            'day_str':       day_str,
            'cost_by_band':  dict(env.cost_by_band),
            'cost_ref_band': dict(env.cost_ref_band),
            'ref_cost':      float(env.ref_cost),
            'pred_kw':       env.pred_kw.tolist(),
        }
    del env
    gc.collect()
    return pd.DataFrame(results), last_env_data

# ──────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE (used by the main pipeline)
# ──────────────────────────────────────────────────────────────────
def evaluate_rule_based(days):
    """Fixed-rule controller used as the comparison baseline.

    - Furnace: 60 % of the measured furnace power during peak hours
      (18-22 h), 100 % otherwise; same thermal model as the environment.
    - Dishwasher: one 60-minute cycle scheduled at step 120 (02:00), using
      the measured dishwasher power when above 0.05 kW and the nominal
      1.5 kW otherwise.
    Returns the same KPI columns as evaluate_multijour.
    """
    results = []
    for day_str in [str(d)[:10] for d in days]:
        try:
            real_kw, _, P_base, P_fur_real, P_dw_real, T_ext_arr, _ = load_day_data(day_str)
        except KeyError:
            continue
        ref_cost   = estimate_ref_cost(real_kw)
        total_cost = 0.0
        discomfort = 0.0
        T_room     = T_INIT
        peaks      = []
        dw_remain  = 0
        dc_obj     = DemandChargeTracker(15)
        cost_bands = {0.08: 0.0, 0.12: 0.0, 0.18: 0.0, 0.30: 0.0}
        ref_bands  = {0.08: 0.0, 0.12: 0.0, 0.18: 0.0, 0.30: 0.0}

        for t in range(len(real_kw)):
            hour  = (t // 60) % 24
            price = get_price(hour)
            dc_r  = 0.6 if (18 <= hour < 22) else 1.0
            P_hvac = dc_r * float(P_fur_real[t])
            T_ext  = float(T_ext_arr[t]) if t < len(T_ext_arr) else 10.0
            dT     = DT / TAU * (-(T_room - T_ext) + R_THERM * P_hvac)
            T_room = float(np.clip(T_room + dT, 10.0, 35.0))

            if t == 120:
                dw_remain = 60

            P_dw_raw    = float(P_dw_real[t])
            P_dw_active = P_dw_raw if P_dw_raw > 0.05 else SmartHomeEnv.P_DW_NOM
            P_dw        = P_dw_active if dw_remain > 0 else 0.0
            if dw_remain > 0:
                dw_remain -= 1

            P_total = max(0.0, float(P_base[t]) + P_hvac + P_dw)
            cost_t  = P_total * price * DT

            total_cost += cost_t
            peaks.append(P_total)
            dc_obj.update(P_total)
            discomfort += (abs(T_room - T_SET) / T_SET) * DT * 60
            cost_bands[price] += cost_t
            ref_bands[price]  += float(real_kw[t]) * price * DT

        results.append({
            'day':              day_str,
            'condition':        'Rule-Based',
            'daily_cost_eur':   total_cost,
            'ref_cost_eur':     ref_cost,
            'cost_reduction_%': (ref_cost - total_cost) / ref_cost * 100.0,
            'peak_kw':          float(np.max(peaks)),
            'peak_before_kw':   float(np.max(real_kw)),
            'discomfort_th':    discomfort,
            'discomfort_dw':    0.0,
            'cost_offpeak':     cost_bands[0.08],
            'cost_shoulder':    cost_bands[0.12],
            'cost_midpeak':     cost_bands[0.18],
            'cost_peak':        cost_bands[0.30],
            'ref_offpeak':      ref_bands[0.08],
            'ref_shoulder':     ref_bands[0.12],
            'ref_midpeak':      ref_bands[0.18],
            'ref_peak':         ref_bands[0.30],
        })
    return pd.DataFrame(results)

# ──────────────────────────────────────────────────────────────────
# MULTI-SEED EXPERIMENT (used by the main pipeline)
# ──────────────────────────────────────────────────────────────────
def run_experiment(train_days, eval_days, seeds=SEEDS, episodes=EPISODES):
    """Train and evaluate one SAC agent per seed against the rule-based baseline.

    - Each seed has its own checkpoint (sac_seed42, sac_seed123, sac_seed7)
      and resumes automatically if that checkpoint exists.
    - After training, the best checkpoint of the seed is reloaded and
      evaluated on all evaluation days, together with the rule-based baseline.
    - The agent with the highest mean cost reduction on the evaluation days
      is saved as sac_best.
    - Per-day KPIs for every seed are written to results_seeds.csv.
    """
    all_results = []
    best_agent  = None
    best_eval   = -np.inf
    all_rewards = []

    for seed in seeds:
        print(f"\n{'='*50}\n  SEED {seed}\n{'='*50}")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        ckpt_path = SAVE_DIR + f"sac_seed{seed}"
        agent     = SACAgent(SmartHomeEnv.STATE_DIM, SmartHomeEnv.ACTION_DIM)

        rewards_hist, best_ep = train_sac_multijour(
            agent, train_days,
            episodes=episodes,
            update_freq=4,
            save_path=ckpt_path,
            patience=10,
            verbose=True,
        )
        all_rewards.append(rewards_hist)

        # Reload the best checkpoint of this seed (not the last episode)
        if os.path.exists(ckpt_path + "_actor.weights.h5"):
            agent.load(ckpt_path)
            print(f"  [EVAL] Meilleur agent seed {seed} rechargé (épisode {best_ep+1})")

        kpis_sac,  _ = evaluate_multijour(agent, eval_days, 'SAC-transformer')
        kpis_rules   = evaluate_rule_based(eval_days)

        for df_kpi, cond in [(kpis_sac, 'SAC-transformer'), (kpis_rules, 'Rule-Based')]:
            df_kpi              = df_kpi.copy()
            df_kpi['seed']      = seed
            df_kpi['condition'] = cond
            all_results.append(df_kpi)

        eval_mean = float(kpis_sac['cost_reduction_%'].mean())
        print(f"  [SEED {seed}] cost_reduction moyen = {eval_mean:.2f}%")

        if eval_mean > best_eval:
            best_eval  = eval_mean
            best_agent = agent
            agent.save(SAVE_DIR + "sac_best")
            print(f"  ✅ Nouveau meilleur global (seed {seed}, réduction {best_eval:.2f}%)")

        gc.collect()

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(SAVE_DIR + "results_seeds.csv", index=False)

    best_seed_idx = int(np.argmax([
        np.mean(r[-5:]) if len(r) >= 5 else np.mean(r) for r in all_rewards
    ]))
    return df_all, best_agent, all_rewards[best_seed_idx]

# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE — training and evaluation (run once; later cells reload
# the saved agent and results)
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("  SAC v9 — transformer | TOUTES CORRECTIONS")
print(f"  Train : {len(train_days)} jours | Eval : {len(eval_days)} jours")
print(f"  Seeds : {SEEDS} | Episodes max : {EPISODES}")
print(f"  STATE_DIM={SmartHomeEnv.STATE_DIM} | ACTION_DIM={SmartHomeEnv.ACTION_DIM}")
print(f"  ALPHA={ALPHA} BETA={BETA} GAMMA={GAMMA}")

df_all, best_agent, rewards_hist = run_experiment(
    train_days, eval_days, seeds=SEEDS, episodes=EPISODES)

summary = (df_all.groupby('condition')[['cost_reduction_%', 'peak_kw', 'discomfort_th']]
                       .agg(['mean', 'std']).round(3))
print("\n=== RÉSULTATS ===")
print(summary.to_string())
summary.to_csv(SAVE_DIR + "results_summary.csv")

# Wilcoxon signed-rank test between SAC and rule-based daily cost reductions.
# Rows are paired by position in df_all (same seed/day order for both).
sac_v = df_all[df_all['condition'] == 'SAC-transformer']
rb_v  = df_all[df_all['condition'] == 'Rule-Based']
n_w   = min(len(sac_v), len(rb_v))
try:
    _, pval = wilcoxon(sac_v['cost_reduction_%'].values[:n_w],
                               rb_v['cost_reduction_%'].values[:n_w])
except Exception:
    pval = float('nan')

# Headline numbers (mean over all seeds and evaluation days)
abstract = {
    'pred_mode':             'transformer',
    'cost_reduction_mean_%': round(float(sac_v['cost_reduction_%'].mean()), 2),
    'cost_reduction_std_%':  round(float(sac_v['cost_reduction_%'].std()),  2),
    'peak_kw_mean':          round(float(sac_v['peak_kw'].mean()),          3),
    'discomfort_th_mean':    round(float(sac_v['discomfort_th'].mean()),    2),
    'wilcoxon_pval':         round(pval, 4),
    'n_seeds':               len(SEEDS),
    'n_eval_days':           len(eval_days),
    'cost_offpeak_sac':      round(float(sac_v['cost_offpeak'].mean()),  4),
    'cost_shoulder_sac':     round(float(sac_v['cost_shoulder'].mean()), 4),
    'cost_midpeak_sac':      round(float(sac_v['cost_midpeak'].mean()),  4),
    'cost_peak_sac':         round(float(sac_v['cost_peak'].mean()),     4),
}

with open(SAVE_DIR + "abstract_metrics_transformer.json", "w") as f:
    json.dump(abstract, f, indent=2)

print("\n" + "=" * 55)
print("  ABSTRACT METRICS [transformer]")
print("=" * 55)
for k, v in abstract.items():
    print(f"  {k:<35}: {v}")
print("=" * 55)

print(f"\n✅ Entraînement terminé — modèle sauvegardé dans {SAVE_DIR}")

# ══════════════════════════════════════════════════════════════════
# FIGURES (IEEE format, PDF and PNG)
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'legend.fontsize':10,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'axes.linewidth': 0.8,
    'lines.linewidth':1.4,
    'axes.grid':      True,
    'grid.alpha':     0.4,
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'grid.color':     '#aaaaaa',
    'savefig.dpi':    300,
    'savefig.bbox':   'tight',
    'savefig.facecolor': 'white',
    'figure.facecolor':  'white',
    'pdf.fonttype':   42,
    'ps.fonttype':    42,
})
# Greyscale/slate palette used in the figures
C_BLUE_FILL   = '#c8d8e8'   # very light blue-grey -> Before (fill)
C_BLUE_LINE   = '#2c5f8a'   # dark blue            -> Before (line)
C_ORANGE_FILL = '#d6d6d6'   # light grey           -> After (fill)
C_ORANGE_LINE = '#444444'   # dark grey/anthracite -> After (line)
C_BLACK       = '#111111'
C_BLUE_DASH   = '#2c5f8a'
C_RED         = '#555555'   # medium grey
C_GREEN       = '#222222'   # near black

# Alternative colour palette (kept for reference)
"""
C_BLUE_FILL   = '#a8c4d4'
C_BLUE_LINE   = '#5B9BD5'
C_ORANGE_FILL = '#f5d9a0'
C_ORANGE_LINE = '#e6a020'
C_BLACK       = '#111111'
C_BLUE_DASH   = '#4a90c4'
C_RED         = '#D62728'
C_GREEN       = '#2CA02C'
"""
IEEE_1COL = (3.5, 2.8)
IEEE_2COL = (7.0, 3.5)

BAND_COLORS = {0.08: '#b0c4d8', # pale blue   -> off-peak
               0.12: '#b8c8b0', # pale green  -> shoulder
               0.18: '#d8cfa0', # pale yellow -> mid-peak
               0.30: '#c8a8a8'} # pale red    -> peak
def _savefig(name: str):
    """Save the current figure as PDF and PNG in SAVE_DIR, show it, then free memory."""
    plt.savefig(SAVE_DIR + f"{name}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(SAVE_DIR + f"{name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    gc.collect()
    print(f"  ✓ {name}.pdf / {name}.png")
def _to_hourly_mean(arr, n):
    """Average a 1-minute daily series into 24 hourly values."""
    sph = max(1, n // 24)
    return np.array([float(np.mean(arr[h*sph:(h+1)*sph]))
                     if len(arr[h*sph:(h+1)*sph]) > 0 else 0.0 for h in range(24)])
def _to_hourly_sum(arr, n):
    """Sum a 1-minute daily series into 24 hourly values."""
    sph = max(1, n // 24)
    return np.array([float(np.sum(arr[h*sph:(h+1)*sph])) for h in range(24)])
def plot_fig_A(env_data: dict):
    """Hourly load before/after control, Transformer forecast, hourly cost and TOU tariff."""
    n = len(env_data['P_before'])
    h_arr    = np.arange(24)
    P_bef_h  = _to_hourly_mean(env_data['P_before'], n)
    P_aft_h  = _to_hourly_mean(env_data['P_total'],  n)
    pred_h   = _to_hourly_mean(env_data['pred_kw'],  n)
    cost_hourly = _to_hourly_sum(env_data['cost_h'], n)

    fig, ax1 = plt.subplots(figsize=IEEE_2COL)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 55))

    tou_hours  = [0, 6, 12, 18, 22, 24]
    tou_prices = [0.08, 0.12, 0.18, 0.30, 0.08]
    tou_x, tou_y = [], []
    for i in range(len(tou_prices)):
        tou_x += [tou_hours[i], tou_hours[i+1]]
        tou_y += [tou_prices[i], tou_prices[i]]
    ax3.fill_between(tou_x, 0, tou_y, alpha=0.10, color='#888888', zorder=0)
    ax3.plot(tou_x, tou_y, color='#777777', lw=1.2, ls='--', label='TOU tariff (€/kWh)', zorder=1)
    ax3.set_ylabel('Tariff (€/kWh)', fontsize=10, color='#666666')
    ax3.set_ylim(0, 0.65)
    ax3.tick_params(axis='y', labelcolor='#666666', labelsize=8)
    ax3.grid(False)

    ax1.fill_between(h_arr, 0, P_aft_h, alpha=0.65, color=C_ORANGE_FILL,
                         label='Consumption with HEMS', zorder=2)
    ax1.fill_between(h_arr, 0, P_bef_h, alpha=0.50, color=C_BLUE_FILL,
                         label='Consumption (Before)', zorder=3)
    ax1.plot(h_arr, P_bef_h, color=C_BLUE_LINE,   lw=1.4, alpha=0.9, zorder=4)
    ax1.plot(h_arr, P_aft_h, color=C_ORANGE_LINE, lw=1.2, alpha=0.8, zorder=3)
    ax1.plot(h_arr, pred_h,  color='#9B59B6', lw=1.0, ls='--', alpha=0.70,
             label='Transformer prediction (kW)', zorder=5)

    ax2.plot(h_arr, cost_hourly, color=C_BLACK, lw=1.8, marker='x', ms=7,
             markeredgewidth=1.8, label='Energy cost (after HEMS)', zorder=6)

    ax1.set_xlabel('Time (Hour)', fontsize=12)
    ax1.set_ylabel('Demand (kW)', fontsize=12)
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel('Hourly Cost (€)', fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax1.legend(h1+h2+h3, l1+l2+l3, loc='upper left', fontsize=8,
               framealpha=0.92, edgecolor='#cccccc', ncol=2)
    plt.tight_layout()
    _savefig("figA_consumption_cost")
def plot_fig_B(env_data: dict):
    """Minute-level total load before and after control, with the P_CAP line and tariff."""
    n     = len(env_data['P_before'])
    hours = np.linspace(0, 24, n)
    pred_arr = np.array(env_data['pred_kw'])

    fig, ax = plt.subplots(figsize=IEEE_2COL)
    ax2 = ax.twinx()

    tou_hours  = [0, 6, 12, 18, 22, 24]
    tou_prices = [0.08, 0.12, 0.18, 0.30, 0.08]
    tou_x, tou_y = [], []
    for i in range(len(tou_prices)):
        tou_x += [tou_hours[i], tou_hours[i+1]]
        tou_y += [tou_prices[i], tou_prices[i]]
    ax2.fill_between(tou_x, 0, tou_y, alpha=0.12, color='#888888',
                     label='TOU tariff (€/kWh)', zorder=0)
    ax2.plot(tou_x, tou_y, color='#555555', lw=1.2, ls='--', zorder=1)
    ax2.set_ylabel('Tariff (€/kWh)', fontsize=11, color='#555555')
    ax2.set_ylim(0, 0.55)
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax2.grid(False)

    ax.fill_between(hours, 0, env_data['P_before'], alpha=0.25, color=C_BLUE_LINE,   zorder=2)
    ax.fill_between(hours, 0, env_data['P_total'],  alpha=0.45, color=C_ORANGE_LINE, zorder=3)
    ax.plot(hours, env_data['P_before'], color=C_BLUE_LINE,   lw=1.4, alpha=0.90,
          label='Consumption (Before)', zorder=4)
    ax.plot(hours, env_data['P_total'],  color=C_ORANGE_LINE, lw=1.4, alpha=0.90,
                  label='Consumption with HEMS', zorder=3)
    ax.axhline(P_CAP, color=C_RED, lw=1.2, ls='--', label=f'Peak cap ({P_CAP} kW)', zorder=5)

    ax.set_xlabel('Time (Hour)', fontsize=12)
    ax.set_ylabel('Power (kW)', fontsize=12)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left', framealpha=0.92, fontsize=8, ncol=2)
    plt.tight_layout()
    _savefig("figB_before_after")
def plot_fig_C(env_data: dict):
    """Dishwasher load before (measured) and after rule-based postponement."""
    P_dw   = np.array(env_data['P_dw'])[:1440]
    P_dw_b = np.array(env_data['P_dw_b'])[:1440]
    n      = len(P_dw)
    hours  = np.linspace(0, 24, n)
    fig, ax = plt.subplots(figsize=IEEE_2COL)
    ax2 = ax.twinx()

    tou_hours  = [0, 6, 12, 18, 22, 24]
    tou_prices = [0.08, 0.12, 0.18, 0.30, 0.08]
    tou_x, tou_y = [], []
    for i in range(len(tou_prices)):
        tou_x += [tou_hours[i], tou_hours[i+1]]
        tou_y += [tou_prices[i], tou_prices[i]]
    ax2.fill_between(tou_x, 0, tou_y, alpha=0.12, color='#888888', zorder=0)
    ax2.plot(tou_x, tou_y, color='#555555', lw=1.2, ls='--', label='TOU tariff (€/kWh)', zorder=1)
    for h, p, lbl in [(3, 0.08, '0.08€'), (9, 0.12, '0.12€'),
                       (15, 0.18, '0.18€'), (20, 0.30, '0.30€')]:
        ax2.text(h, p+0.01, lbl, fontsize=7, color='#444444', ha='center')
    ax2.set_ylabel('Tariff (€/kWh)', fontsize=11, color='#555555')
    ax2.set_ylim(0, 0.55)
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax2.grid(False)

    ax.fill_between(hours, 0, P_dw_b, alpha=0.35, color=C_BLUE_FILL,
    label='DW – Before SAC (measured, kW)', zorder=2)
    ax.plot(hours, P_dw_b, color=C_BLUE_LINE, lw=1.2, alpha=0.8, zorder=3)
    ax.fill_between(hours, 0, P_dw, alpha=0.55, color=C_ORANGE_FILL,
    label='DW – After SAC (shifted to off-peak, kW)', zorder=2)
    ax.plot(hours, P_dw, color=C_ORANGE_LINE, lw=1.2, zorder=3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8, framealpha=0.92, loc='upper right', ncol=2)
    ax.set_xlabel('Time (hour)', fontsize=12)
    ax.set_ylabel('DW Power (kW)', fontsize=12)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(bottom=-0.05)
    plt.tight_layout()
    _savefig("figC_dishwasher")
def plot_fig_D(env_data: dict, zoom_start_h: float = None, zoom_end_h: float = None):
    """Furnace load before/after control: a zoomed window and the full 24 h.

    Without an explicit window, the 3-hour window with the most furnace
    activity is chosen automatically.
    """
    n       = len(env_data['P_fur'])
    hours   = np.linspace(0, 24, n, endpoint=False)
    P_fur_b   = np.array(env_data['P_fur_b'])
    P_fur_aft = np.array(env_data['P_fur'])

    if zoom_start_h is None or zoom_end_h is None:
        window = 180
        if n > window:
            activity   = np.convolve(P_fur_b, np.ones(window), mode='valid')
            best_start = int(np.argmax(activity))
        else:
            best_start = 0
        h_start = np.floor(hours[best_start] * 2) / 2
        h_end   = min(24.0, np.ceil((hours[best_start] + 3.0) * 2) / 2)
        if zoom_start_h is None: zoom_start_h = h_start
        if zoom_end_h   is None: zoom_end_h   = h_end

    mask   = (hours >= zoom_start_h) & (hours < zoom_end_h)
    h_zoom = hours[mask]
    if len(h_zoom) == 0:
        print(f"  [WARN] plot_fig_D : aucun point dans [{zoom_start_h}, {zoom_end_h}]h")
        return

    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(h_zoom, P_fur_b[mask],   color=C_BLUE_DASH,   lw=1.4, ls='--', label='Furnace – Before')
    ax.plot(h_zoom, P_fur_aft[mask], color=C_ORANGE_LINE, lw=1.4, ls='-', label='Furnace – After (DRL)')
    ax.set_xlabel('Time (Hour)', fontsize=12)
    ax.set_ylabel('Power (kW)', fontsize=12)
    ax.set_xlim(zoom_start_h, zoom_end_h)
    tick_step = 0.5
    ticks = np.arange(zoom_start_h, zoom_end_h + tick_step/2, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(t)}' if t == int(t) else f'{t:.1f}' for t in ticks],
                           fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.92,
              edgecolor='#cccccc', handlelength=2.0)
    plt.tight_layout()
    _savefig("figD_furnace_zoom")

    fig2, ax2 = plt.subplots(figsize=IEEE_2COL)
    ax2.plot(hours, P_fur_b,   color=C_BLUE_DASH,   lw=1.0, ls='--', label='Furnace – Before')
    ax2.plot(hours, P_fur_aft, color=C_ORANGE_LINE, lw=1.0, ls='-',  label='Furnace – After (DRL)')
    ax2.set_xlabel('Time (Hour)', fontsize=12)
    ax2.set_ylabel('Power (kW)', fontsize=12)
    ax2.set_xlim(0, 24)
    ax2.set_xticks([0, 6, 12, 18, 24])
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.92, handlelength=2.0)
    plt.tight_layout()
    _savefig("figD_furnace_full24h")
def plot_fig_E(df_all):
    """Stacked mean daily cost per tariff band: baseline, rule-based and SAC."""
    conditions = [c for c in ['Rule-Based', 'SAC-transformer']
                  if c in df_all['condition'].unique()]
    base_rows = []
    ref_cond  = conditions[0]
    for _, row in df_all[df_all['condition'] == ref_cond].iterrows():
        base_rows.append({
            'condition':     'Baseline',
            'cost_offpeak':  row.get('ref_offpeak',  0),
            'cost_shoulder': row.get('ref_shoulder', 0),
            'cost_midpeak':  row.get('ref_midpeak',  0),
            'cost_peak':     row.get('ref_peak',     0),
        })
    df_base = pd.DataFrame(base_rows)
    df_plot  = pd.concat([df_base, df_all[df_all['condition'].isin(conditions)]],
                           ignore_index=True)
    all_conds = ['Baseline'] + conditions

    bands = [
        ('cost_offpeak',  'Off-peak  0.08€/kWh', '#2A9D8F'),
        ('cost_shoulder', 'Shoulder  0.12€/kWh', '#7B2D8B'),
        ('cost_midpeak',  'Mid-peak  0.18€/kWh', '#C9A227'),
        ('cost_peak',     'Peak      0.30€/kWh', '#0000FF'),
    ]
    x       = np.arange(len(all_conds))
    fig, ax = plt.subplots(figsize=IEEE_2COL)
    bottoms = np.zeros(len(all_conds))

    for col, label, color in bands:
        vals = np.array([
            float(df_plot[df_plot['condition'] == c][col].mean())
            if len(df_plot[df_plot['condition'] == c]) > 0 else 0.0
            for c in all_conds
        ])
        bars = ax.bar(x, vals, width=0.55, bottom=bottoms,
                      label=label, color=color, alpha=0.88,
                      edgecolor='white', linewidth=0.8, zorder=3)
        for bar, v, bot in zip(bars, vals, bottoms):
            if v > 0.04:
                ax.text(bar.get_x() + bar.get_width()/2, bot + v/2,
                        f'{v:.3f}€', ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold')
        bottoms += vals

    for xi, tot in zip(x, bottoms):
        ax.text(xi, tot+0.02, f'{tot:.3f}€',
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=C_BLACK)

    ax.plot(x, bottoms, color=C_BLACK, lw=1.5, marker='D', ms=6,
            markerfacecolor='white', markeredgewidth=1.5,
            zorder=6, label='Total daily cost (€)')

    if len(bottoms) >= 3:
        red_pct = (bottoms[0] - bottoms[2]) / bottoms[0] * 100
        ax.annotate(f'−{red_pct:.1f}%',
                    xy=(x[2], bottoms[2]), xytext=(x[2]+0.3, bottoms[2]+0.15),
                    fontsize=9, color=C_GREEN, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(['Uncontrolled\n(Baseline)',
                        'Rule-Based\nControl',
                        'SAC + Transformer\n(Proposed)'], fontsize=10)
    ax.set_ylabel('Daily Electricity Cost (€)', fontsize=12)
    ax.set_xlabel('Control Strategy', fontsize=12)
    ax.set_ylim(0, bottoms.max() * 1.20)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.92, ncol=2,
              edgecolor='#cccccc', handlelength=1.0, handletextpad=0.4,
              columnspacing=0.6, borderpad=0.4, labelspacing=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _savefig("figE_cost_by_tariff")
def plot_fig_F(rewards_hist):
    """Learning curve of one seed (performance = -episode reward) with rolling mean."""
    perf = -np.array(rewards_hist)
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(perf, color=C_BLUE_LINE, lw=0.7, alpha=0.4, label='Episode performance')
    w    = max(3, len(perf) // 5)
    roll = pd.Series(perf).rolling(w, min_periods=1).mean()
    ax.plot(roll, color=C_BLACK, lw=1.4, label=f'Rolling mean (w={w})')
    if float(roll.iloc[-1]) < float(roll.iloc[0]):
        ax.text(0.97, 0.08, '⚠ Performance decreasing\nReduce lr to 1e-4',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                color=C_RED, bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='#fff0f0', edgecolor=C_RED, alpha=0.9))
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Performance (−reward)', fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    _savefig("figF_learning_curve")



# ══════════════════════════════════════════════════════
# 48-HOUR DISHWASHER PLOT — cycle postponed to the next day
# Uses `best_agent` from the main pipeline above.
# ══════════════════════════════════════════════════════

# 1. Find the first evaluation day with dishwasher activity after 18:00
def find_dw_peak_day(eval_days):
    for day_str in eval_days:
        try:
            real_kw, pred_kw, P_base, P_fur_real, P_dw_real, T_ext_arr, dw_cycle_len = load_day_data(day_str)
            steps_18h = int(18 * 60)
            if P_dw_real[steps_18h:].max() > 0.1:
                return day_str
        except:
            continue
    return None

day1 = find_dw_peak_day(eval_days)
day2 = eval_days[list(eval_days).index(day1) + 1]  # following evaluation day
print(f"Jour 1 : {day1}  |  Jour 2 : {day2}")

# 2. Simulate both days with the trained agent. Each day is an independent
#    episode, so a cycle postponed on day 1 is not carried over to day 2.
def simulate_day(agent, day_str):
    """Run one day deterministically and return dishwasher and price traces."""
    env   = make_env(day_str)
    state = env.reset()
    while True:
        action          = agent.select_action(state, deterministic=True)
        ns, _, done, _  = env.step(action)
        state           = ns
        if done:
            break
    return {
        'day_str' : day_str,
        'P_dw_b'  : np.array(env.hist_P_dw_b),
        'P_dw'    : np.array(env.hist_P_dw),
        'price'   : np.array(env.hist_price),
    }

ed1 = simulate_day(best_agent, day1)
ed2 = simulate_day(best_agent, day2)

# 3. Plot the 48 hours (displayed only, not saved to disk)
def plot_dw_48h(ed1, ed2):
    """Dishwasher load before/after control over two consecutive days."""
    dw_b   = np.concatenate([ed1['P_dw_b'], ed2['P_dw_b']])
    dw_a   = np.concatenate([ed1['P_dw'],   ed2['P_dw']])
    prices = np.concatenate([ed1['price'],  ed2['price']])
    n      = len(dw_b)
    hours  = np.linspace(0, 48, n, endpoint=False)

    fig, ax1 = plt.subplots(figsize=(15, 4))
    ax2 = ax1.twinx()

    # Tariff bands as background shading
    bands_48 = [
        (0,6,'#5B9BD5'), (6,12,'#70AD47'), (12,18,'#FFC000'),
        (18,22,'#FF4444'), (22,24,'#5B9BD5'),
        (24,30,'#5B9BD5'), (30,36,'#70AD47'), (36,42,'#FFC000'),
        (42,46,'#FF4444'), (46,48,'#5B9BD5')
    ]
    for h0, h1, col in bands_48:
        ax1.axvspan(h0, h1, alpha=0.08, color=col)

    ax1.fill_between(hours, 0, dw_b, alpha=0.20, color='steelblue')
    ax1.fill_between(hours, 0, dw_a, alpha=0.45, color='darkorange')
    ax1.plot(hours, dw_b, color='steelblue',  lw=1.2, ls='--', label='DW – Before SAC')
    ax1.plot(hours, dw_a, color='darkorange', lw=1.5,          label='DW – After SAC (déplacé)')
    ax2.step(hours, prices, color='gray', lw=1.0, ls=':', where='post', label='Tariff (€/kWh)')
    ax1.axvline(24, color='black', lw=1.8, ls='--', alpha=0.6, label='Minuit → Jour 2')

    # Annotations
    ax1.annotate('DW reporté\nau lendemain',
                    xy=(25, dw_a[int(25/48*n)]),
                    xytext=(27, 0.8),
                    arrowprops=dict(arrowstyle='->', color='darkorange'),
                    fontsize=9, color='darkorange')

    ax1.set_xlabel('Time (Hour)', fontsize=11)
    ax1.set_ylabel('DW Power (kW)', fontsize=11)
    ax2.set_ylabel('Tariff (€/kWh)', fontsize=10)
    ax1.set_xlim(0, 48)
    ax1.set_xticks(range(0, 49, 2))
    ax1.set_xticklabels([f'{h%24}h' + ('\n◄J2' if h == 24 else '') for h in range(0, 49, 2)], fontsize=8)
    ax1.set_ylim(bottom=0)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, fontsize=9, loc='upper center')
    ax1.set_title(
        f"Dishwasher 48h — DW déplacé vers off-peak lendemain\n"
        f"Jour 1 : {ed1['day_str']}  →  Jour 2 : {ed2['day_str']}", fontsize=11)
    plt.tight_layout()
    plt.show()

plot_dw_48h(ed1, ed2)

# ══════════════════════════════════════════════════════════════════
# FIGURE CELL — reloads the saved agent and results; no retraining
# ══════════════════════════════════════════════════════════════════

# ── 1. Load the best agent ──
best_path = SAVE_DIR + "sac_best"
if not os.path.exists(best_path + "_actor.weights.h5"):
    raise FileNotFoundError(f"❌ Aucun agent sauvegardé dans : {best_path}\n"
                            "   Lance d'abord le PIPELINE PRINCIPAL.")

best_agent = SACAgent(SmartHomeEnv.STATE_DIM, SmartHomeEnv.ACTION_DIM)
best_agent.load(best_path)
print(f"✅ Agent chargé depuis : {best_path}")

# ── 2. Load per-day results (recomputed with seed 42 label if missing) ──
results_csv = SAVE_DIR + "results_seeds.csv"
if os.path.exists(results_csv):
    df_all = pd.read_csv(results_csv)
    print(f"✅ Résultats chargés : {results_csv}  ({len(df_all)} lignes)")
else:
    print("⚠️  results_seeds.csv absent — recalcul des KPIs...")
    kpis_sac = evaluate_multijour(best_agent, eval_days, 'SAC-transformer')[0]
    kpis_rules = evaluate_rule_based(eval_days)
    kpis_sac['seed'] = 42
    kpis_rules['seed'] = 42
    df_all = pd.concat([kpis_sac, kpis_rules], ignore_index=True)

# ── 3. Day shown in the per-day figures. The first assignment is overridden
#       by the fixed date on the next line; edit DATE_DISPLAY to change it. ──
DATE_DISPLAY = eval_days[2]

DATE_DISPLAY = '2016-12-10'

# Fall back to the first evaluation day if that date is not available
try:
    load_day_data(str(DATE_DISPLAY)[:10])
    print(f"✅ Journée valide : {DATE_DISPLAY}")
except (KeyError, ValueError):
    print(f"❌ {DATE_DISPLAY} absent — utilisation de eval_days[0]")
    DATE_DISPLAY = eval_days[0]
    print(f"   → Remplacement par : {DATE_DISPLAY}")

# ── 4. Simulate the chosen day and keep its trajectories ──
_, env_data = evaluate_multijour(
    best_agent, [str(DATE_DISPLAY)[:10]], 'SAC-transformer', keep_last_env=True)

if env_data is None:
    raise ValueError(f"❌ env_data est None pour {DATE_DISPLAY}")

print(f"✅ env_data généré pour : {env_data['day_str']}")
# ── 5. Figures A-E. Fig. F is disabled (wrapped in a string literal below);
#       it needs the reward history from a training run in the same session. ──
print("\n── Génération des figures ──")
plot_fig_A(env_data)
print("✅ Fig A")
plot_fig_B(env_data)
print("✅ Fig B")
plot_fig_C(env_data)
print("✅ Fig C")
plot_fig_D(env_data, zoom_start_h=18.0, zoom_end_h=22.0)
print("✅ Fig D")
plot_fig_E(df_all)
print("✅ Fig E")
"""
plot_fig_F(rewards_hist if 'rewards_hist' in dir() else []) ####### je lance avec entrainemment
print("✅ Fig F")"""

# ── 6. Headline metrics recomputed from df_all, with a one-sample Wilcoxon
#       signed-rank test of the SAC cost reductions. ──
sac_df = df_all[df_all['condition'] == 'SAC-transformer']
try:
    _, pval = wilcoxon(sac_df['cost_reduction_%'])
except Exception:
    pval = float('nan')

print("\n" + "="*55)
print("  ABSTRACT METRICS [transformer]")
print("="*55)
print(f"  {'cost_reduction_mean_%':<35}: {sac_df['cost_reduction_%'].mean():.2f}")
print(f"  {'cost_reduction_std_%':<35}: {sac_df['cost_reduction_%'].std():.2f}")
print(f"  {'peak_kw_mean':<35}: {sac_df['peak_kw'].mean():.3f}")
print(f"  {'discomfort_th_mean':<35}: {sac_df['discomfort_th'].mean():.2f}")
print(f"  {'wilcoxon_pval':<35}: {pval:.4f}")
print(f"  {'n_eval_days':<35}: {len(sac_df)}")
print(f"  {'cost_offpeak_sac':<35}: {sac_df['cost_offpeak'].mean():.4f}")
print(f"  {'cost_shoulder_sac':<35}: {sac_df['cost_shoulder'].mean():.4f}")
print(f"  {'cost_midpeak_sac':<35}: {sac_df['cost_midpeak'].mean():.4f}")
print(f"  {'cost_peak_sac':<35}: {sac_df['cost_peak'].mean():.4f}")
print("="*55)
# Peak-to-average ratio (PAR) before and after control
sac_df = df_all[df_all['condition'] == 'SAC-transformer'].copy()

sac_df['PAR_after']  = sac_df['peak_kw']        / sac_df['mean_kw']
sac_df['PAR_before'] = sac_df['peak_before_kw'] / sac_df['mean_before_kw']

PAR_before    = sac_df['PAR_before'].mean()
PAR_after     = sac_df['PAR_after'].mean()
PAR_reduction = (PAR_before - PAR_after) / PAR_before * 100

print("\n" + "="*55)
print("  PAR METRICS")
print("="*55)
print(f"  PAR Before    : {PAR_before:.3f}")
print(f"  PAR After SAC : {PAR_after:.3f}")
print(f"  PAR reduction : {PAR_reduction:.1f}%")
print("="*55)

gc.collect()
print("\n✅ Affichage terminé — aucun réentraînement effectué")

# FURNACE LOAD WITH INDOOR TEMPERATURE


def plot_fig_D_temp(env_data: dict):
    """Furnace load before/after control over 24 h, with hourly indoor temperature."""
    n         = len(env_data['P_fur'])
    hours     = np.linspace(0, 24, n, endpoint=False)
    P_fur_b   = np.array(env_data['P_fur_b'])
    P_fur_aft = np.array(env_data['P_fur'])
    T_room    = np.array(env_data.get('T_room', []))

    fig, ax_p = plt.subplots(figsize=IEEE_2COL)
    ax_t = ax_p.twinx()

    # Furnace load
    ax_p.plot(hours, P_fur_b,   color=C_BLUE_DASH,   lw=1.0, ls='--', label='Furnace – Before')
    ax_p.plot(hours, P_fur_aft, color=C_ORANGE_LINE, lw=1.0, ls='-', label='Furnace – After (DRL)')

    # Indoor temperature (hourly mean) and set-point
    if len(T_room) == n:
        T_h = _to_hourly_mean(T_room, n)
        ax_t.plot(np.arange(24), T_h, color='#D62728', lw=1.4, ls='-.', marker='o', ms=3, label='Indoor Temp (°C)')
        ax_t.axhline(T_SET, color='#D62728', lw=0.8, ls=':', alpha=0.6, label=f'Setpoint ({T_SET}°C)')
        ax_t.set_ylabel('Temperature (°C)', fontsize=11, color='#D62728')
        ax_t.tick_params(axis='y', labelcolor='#D62728')
        ax_t.set_ylim(T_SET - 3, T_SET + 3)

    # TOU bands as background shading
    for h0, h1, col, _, _ in TARIF_BANDS:
        ax_p.axvspan(h0, h1, alpha=0.06, color=col)

    ax_p.set_xlabel('Time (Hour)', fontsize=12)
    ax_p.set_ylabel('Power (kW)', fontsize=12)
    ax_p.set_xlim(0, 24)
    ax_p.set_xticks([0, 6, 12, 18, 24])
    ax_p.set_ylim(bottom=0)

    lines1, labs1 = ax_p.get_legend_handles_labels()
    lines2, labs2 = ax_t.get_legend_handles_labels()
    ax_p.legend(lines1 + lines2, labs1 + labs2, loc='upper left', fontsize=9, framealpha=0.92)
    plt.tight_layout()
    _savefig("figD_furnace_temp")

plot_fig_D_temp(env_data)
print("✅ Fig D + Température")



# CONVERTING THE THERMAL DISCOMFORT INDEX INTO A MEAN TEMPERATURE DEVIATION
# discomfort_th = sum over the day's minutes of |T_room - T_SET| / T_SET, so
# the mean deviation in °C is discomfort_th / 1440 * T_SET.
# Worked example (notes below): 78.94 / 1440 * 22 = 1.206 °C.
"""self._disc_th_acc += rho_f * DT * 60
# où rho_f = abs(T_room - T_SET) / T_SET
# DT = 1/60 heure → DT * 60 = 1 minutes
78.94 = Σ |T_room - 22| / 22   (en minutes sur toute la journée)
# Écart moyen en °C sur la journée
T_ecart_moyen = (78.94 / 1440) * T_SET
print(f"Écart moyen : {T_ecart_moyen:.3f} °C")
# → 78.94 / 1440 * 22 = 1.206 °C"""

# LEARNING CURVE OVER THE THREE SEEDS (from the saved reward histories)
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

seeds = [42, 7, 123]
all_rewards = []

for s in seeds:
    with open(SAVE_DIR + f"sac_seed{s}_rewards_hist.json", 'r') as f:
        all_rewards.append(json.load(f)['rewards'])

# Truncate to the shortest history (seeds may stop early at different episodes)
min_len = min(len(r) for r in all_rewards)
arr = np.array([r[:min_len] for r in all_rewards])  # shape (3, episodes)

mean_r = arr.mean(axis=0)
std_r  = arr.std(axis=0)
perf   = -mean_r
perf_std = std_r
episodes = np.arange(min_len)

fig, ax = plt.subplots(figsize=IEEE_2COL)

# Individual seeds in light grey
for i, s in enumerate(seeds):
    ax.plot(-arr[i], color='#aaaaaa', lw=0.6, alpha=0.4)

# Rolling mean across seeds
w    = max(3, min_len // 5)
roll = pd.Series(perf).rolling(w, min_periods=1).mean()
ax.plot(roll, color=C_BLACK, lw=1.8, label=f'Mean (3 seeds), rolling w={w}')

# ±1 standard deviation across seeds (rolling)
roll_std = pd.Series(perf_std).rolling(w, min_periods=1).mean()
ax.fill_between(episodes,
                roll - roll_std,
                roll + roll_std,
                alpha=0.15, color=C_BLACK, label='±1 std')

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Performance (−reward)', fontsize=12)
ax.legend(fontsize=9)
ax.set_title('SAC Learning Curve — 3 Seeds')
plt.tight_layout()
_savefig("figF_learning_curve_3seeds")

# INSPECTION: list the output folder and peek at a saved reward history

import os
print("Fichiers dans SAVE_DIR :")
for f in sorted(os.listdir(SAVE_DIR)):
    print(f"  {f}")

import json
with open(SAVE_DIR + "sac_seed42_rewards_hist.json", 'r') as f:
    data = json.load(f)

print(type(data))
print(str(data)[:300])

# Columns available in the results table
print(df_all.columns.tolist())