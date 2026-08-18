# Forecasting the IDR/USD Exchange Rate Using an ARIMAX-GARCH Model

Replication code and data for the paper *"Forecasting the IDR/USD Exchange Rate
Using an ARIMAX-GARCH Model with Macroeconomic Exogenous Variables"* (Lobo, Syam,
& Sanusi). The study forecasts the daily Indonesian rupiah to US dollar (IDR/USD)
exchange rate with an ARIMAX(2)-GARCH(1,1) model and evaluates it, under a
leakage-free fixed-size rolling window, against three benchmarks: a driftless
random walk, a random walk with drift, and a nested ARIMA-GARCH control.

## Repository contents

| File | Description |
|------|-------------|
| `Forecasting IDR-USD with ARIMAX-GARCH.ipynb` | Complete analysis notebook: preprocessing, unit-root tests (Table 2), model estimation and diagnostics (Table 3, Section 4.4), leakage-free rolling-window forecasts for the four models, Diebold-Mariano tests with the HLN correction (Table 4), the random-walk-with-drift robustness check, and Figures 1-4. |
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
   jupyter notebook "Forecasting IDR-USD with ARIMAX-GARCH.ipynb"
   ```

   The notebook reads `merged_data_clean.csv` from the same folder and reproduces
   Tables 2-4, the robustness check, and Figures 1-4.

All forecasts are strictly out-of-sample: at each rolling origin the model is
re-estimated using only past observations, and the exogenous regressors are held
at their last observed level over the forecast horizon, so no future information
enters any forecast.

## Citation

If you use this code or data, please cite:

> Lobo, S. Y., Syam, R., & Sanusi, W. (2026). Forecasting the IDR/USD exchange
> rate using an ARIMAX-GARCH model with macroeconomic exogenous variables.
> [Journal name, volume(issue), pages.]
