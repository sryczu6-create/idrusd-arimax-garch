# Forecasting the Rupiah Exchange Rate Against a Drift-Adjusted Benchmark

Replication code and data for the paper *"Forecasting the Rupiah Exchange Rate
Against a Drift-Adjusted Benchmark"* (Lobo, Syam, & Sanusi). The study forecasts
the daily Indonesian rupiah to US dollar (IDR/USD) exchange rate with an
ARIMAX(2)-GARCH(1,1) model and evaluates it, under a leakage-free fixed-size
rolling window, against three benchmarks: a driftless random walk, a random walk
with drift, and a nested ARIMA-GARCH control.

## Repository contents

| File | Description |
|------|-------------|
| `Forecasting_IDR_USD_with_ARIMAX_GARCH.ipynb` | Complete analysis notebook: preprocessing, unit-root tests (Table II), Johansen cointegration test, model estimation and diagnostics (Table III), leakage-free rolling-window forecasts for the four models, Diebold-Mariano tests with the HLN correction (Tables IV and V), the random-walk-with-drift robustness check, and Figures 1-4. |
| `merged_data_clean.csv` | Daily aligned dataset: `date`, `fx` (JISDOR IDR/USD), `ffr` (Federal Funds Rate), `bi` (BI Rate), `cpi` (Indonesian CPI). |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | License for the code. |

## Data sources

The daily IDR/USD rate is the Jakarta Interbank Spot Dollar Rate (JISDOR)
published by Bank Indonesia. The Federal Funds Rate is from the Federal Reserve
Economic Data (FRED) service (series DFEDTARU). The Indonesian Consumer Price
Index is reconstructed from Statistics Indonesia (BPS) monthly inflation
releases. The BI Rate is compiled from Bank Indonesia announcements. The sample
covers 2 April 2018 to 27 July 2026. Exogenous series are aligned to the daily
frequency by last-observation-carried-forward, as described in the paper.

## How to reproduce

1. Install dependencies (Python 3.11 recommended):

   ```
   pip install -r requirements.txt
   ```

2. Open the notebook and run all cells top to bottom (Jupyter, or Google Colab):

   ```
   jupyter notebook "Forecasting_IDR_USD_with_ARIMAX_GARCH.ipynb"
   ```

   The notebook reads `merged_data_clean.csv` from the same folder and reproduces
   Tables II-V, the Johansen cointegration test, the robustness check, and
   Figures 1-4.

All forecasts are strictly out-of-sample: at each rolling origin the model is
re-estimated using only past observations, and the exogenous regressors are held
at their last observed level over the forecast horizon, so no future information
enters any forecast.
