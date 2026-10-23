# Experiments

The experiments are four Python scripts (`../scripts/`), one per Google
Colab notebook. They exchange data through a single output folder,
`SAVE_DIR` (default `/content/drive/MyDrive/predictionsfinal/`), and must be
run **in order**.

```
HomeC.csv.zip
     │
     ▼
01_lstm_forecasting.py ──► scaler_lstm.pkl, lstm_pred.csv, lstm_val_pred.csv
     │
     ▼
02_transformer_forecasting.py ──► trans_pred.csv, trans_val_pred.csv
     │                             (loads scaler_lstm.pkl)
     ├──► (optional) rerun 01 to draw the Transformer figures and the
     │    LSTM/Transformer comparison table
     ▼
03_lstm_transformer_ensemble.py ──► ensemble_pred.csv, metrics_4models.csv
     │
     ▼
04_sac_energy_management.py ──► SAC agents, results_seeds.csv, figures
        (loads trans_pred.csv, scaler_lstm.pkl and HomeC.csv.zip)
```

## Experiment 1: one-step-ahead load forecasting (scripts 01, 02)

- **Task:** predict whole-house load (`House overall`, kW) one minute ahead
  from the previous 60 minutes of five features.
- **Split:** chronological 70 / 15 / 15 (train / validation / test).
- **Models:** persistence, LSTM, Transformer encoder (see `../models/README.md`).
- **Metrics (test set):** MAE, RMSE and R² on normalised values; MAE and RMSE
  in kW; MAPE (%) in kW, excluding minutes with measured load ≤ 0.05 kW;
  relative improvement over persistence.
- **Sensitivity analysis:** Gaussian noise with σ ∈ {0, 0.001, 0.005, 0.01,
  0.02, 0.05} (normalised units) is added to the test predictions, 10 draws
  each; the analysis reports the RMSE increase.

| Output (01) | Output (02) |
|---|---|
| `lstm_pred.csv`, `lstm_val_pred.csv` | `trans_pred.csv`, `trans_val_pred.csv` |
| `lstm_history.json` | `transformer_history.json` |
| `metrics_lstm_full.csv` | `metrics_transformer_full.csv` |
| `sensitivity_analysis.csv` | `sensitivity_analysis_transformer.csv`, `sensitivity_comparison_lstm_tr.csv`¹ |
| `ieee_lstm_loss`, `ieee_lstm_short_term`, `ieee_lstm_scatter_error` (.pdf/.png) | `ieee_trans_loss`, `ieee_trans_short_term`, `ieee_trans_scatter_error` (.pdf/.png) |
| `ieee_comparison_full.csv` and Transformer figures² | `TABLE_IX_tr_stats.csv` |
| `computational_stats_all_models.csv`, `TABLE_IX_computational_statistics.csv` | |

¹ only if `sensitivity_analysis.csv` from 01 exists.
² only if `trans_pred.csv` from 02 already exists.

Prediction CSVs contain a `time` index, the scaled actual value (`real`) and
the scaled prediction (`pred_lstm` / `pred_transformer`).

## Experiment 2: LSTM–Transformer ensemble (script 03)

- The weight w is selected on the **validation** predictions, then applied
  unchanged to the test predictions.
- Without validation predictions, the two models are weighted equally
  (w = 0.5).
- **Outputs:** `ensemble_weight.txt`, `ensemble_pred.csv`, `metrics_4models.csv`
  (persistence, LSTM, Transformer, ensemble), figures
  `fig1_ensemble_weight_search`, `fig2_error_distribution`,
  `fig3_metrics_barplot`, `fig4_short_term_kw` (.pdf/.png).

## Experiment 3: forecast-informed energy management with SAC (script 04)

- **Episode:** one day at a 1-minute step.
- **Controlled loads:** furnace (duty cycle chosen by the agent, capped in
  peak/mid-peak hours) and dishwasher (rule-based postponement to off-peak).
- **Thermal model:** first-order RC (R = 15, C = 50, set-point 22 °C,
  initial 21 °C).
- **Tariff (€/kWh):** 0.08 (0–6 h, 22–24 h), 0.12 (6–12 h), 0.18 (12–18 h),
  0.30 (18–22 h).
- **Reward:** −(0.40 · cost/0.002 + 0.25 · 25 · excess-over-3.5 kW/3.5
  + 0.35 · discomfort/0.045).
- **Days:** evaluation on every day covered by `trans_pred.csv`; training
  on up to 20 other days (one per ISO week).
- **Seeds:** 42, 123, 7; at most 50 episodes each with early stopping.
- **Baselines:** uncontrolled measured consumption and a rule-based
  controller.
- **KPIs per day:** daily cost and cost reduction (%), peak power before and
  after control, thermal and dishwasher discomfort, cost per tariff band.
  The script also runs a Wilcoxon signed-rank test and computes the
  peak-to-average ratio.
- **Outputs:** `sac_seed*_rewards_hist.json`, `results_seeds.csv`,
  `results_summary.csv`, `abstract_metrics_transformer.json`, figures
  `figA_consumption_cost`, `figB_before_after`, `figC_dishwasher`,
  `figD_furnace_zoom`, `figD_furnace_full24h`, `figE_cost_by_tariff`,
  `figD_furnace_temp`, `figF_learning_curve_3seeds` (.pdf/.png).
