# Data

## 1. Dataset

The experiments use the **UMass Smart\* Data Set for Sustainability** (Smart\*
Home data, home "C"), distributed through the UMass Trace Repository:

**https://traces.cs.umass.edu/docs/traces/smartstar/**

The Smart\* project collects detailed measurements from real homes: whole-house
and per-circuit electricity consumption, plug loads, on-site generation,
outdoor weather, and indoor environmental data.

Reference publication for the dataset:

> S. Barker, A. Mishra, D. Irwin, E. Cecchet, P. Shenoy and J. Albrecht,
> "Smart\*: An Open Data Set and Tools for Enabling Research in Sustainable
> Homes," in *Proc. KDD Workshop on Data Mining Applications in
> Sustainability (SustKDD)*, 2012.

Access and use are subject to the terms published on the UMass Trace
Repository. The data are therefore obtained directly from that source and are
not stored in this repository.

## 2. File used by the scripts

| Item | Value |
|---|---|
| File | `HomeC.csv.zip`: a zip archive containing `HomeC.csv` (pandas reads it directly) |
| Content | minute-level electrical measurements for home C together with weather variables, in one CSV file with a header row |
| Sampling | 1 minute |
| Time axis | regular 1-minute index starting on 2016-01-01 at 05:00, built by the scripts row by row |
| Power units | kW; column names carry a ` [kW]` suffix (e.g. `House overall [kW]`), which the scripts remove |

## 3. Variables

### Load forecasting (scripts 01–03)

| Role | Column | Unit |
|---|---|---|
| Target | `House overall` | kW |
| Input: lagged target | `House overall` | kW |
| Input: weather | `temperature`, `precipIntensity` | as provided in the file |
| Input: calendar | `hour` (0–23), `dayofweek` (0 = Monday) | derived from the time index |

### Energy management (script 04)

| Use | Column(s) |
|---|---|
| Measured total load | `House overall` |
| Controllable thermal load | `Furnace 1` + `Furnace 2` |
| Shiftable load | `Dishwasher` |
| Non-controllable base load | `House overall` − furnace − dishwasher (clipped at 0) |
| Outdoor temperature | `temperature`, converted from °F to °C for the thermal model |

## 4. Preprocessing

1. Read the zipped CSV with pandas.
2. Remove the ` [kW]` suffix from the column names.
3. Assign a regular 1-minute `DatetimeIndex` starting at 2016-01-01 05:00.
4. Derive `month`, `hour` and `dayofweek` from the index.
5. Replace ±inf with NaN and drop incomplete rows.
6. Select the five forecasting inputs and split them chronologically:
   70 % training, 15 % validation, 15 % test.
7. Fit a Min-Max scaler (range [0, 1]) on the training split and apply it to
   the validation and test splits. The same scaler (`scaler_lstm.pkl`) is
   shared by the LSTM, Transformer, ensemble and SAC scripts.
8. Build sliding windows of 60 consecutive minutes (5 features) to predict
   the scaled `House overall` value at the next minute. Windows are built
   within each split.

For the energy-management experiment, the **evaluation days** are the
calendar days covered by the Transformer test predictions. The **training
days** are the remaining days of the file, one per ISO week, up to 20 days.

## 5. Placing the data

**Google Colab (default configuration of the scripts)**

1. Download the data from the UMass Trace Repository.
2. Save `HomeC.csv.zip` in a Google Drive folder named `SMART HOME DATA` at
   the root of *My Drive*. The scripts expect
   `/content/drive/MyDrive/SMART HOME DATA/HomeC.csv.zip`.
3. Run the scripts in Colab; they mount Google Drive automatically.

**Local machine**

1. Save the file as `data/HomeC.csv.zip` in this repository. The `data/`
   folder is excluded from version control by `.gitignore`.
2. Set the data and output paths at the top of each script, as described in
   the main `README.md` (section 5).
