# Leakage-free rolling-window evaluation of an ARIMAX(2)-GARCH(1,1) model for the IDR/USD exchange rate

Code and data for *"Rolling-Window Benchmarking of Macroeconomic Models
for Rupiah Exchange Rate Forecasting."* The analysis forecasts the daily JISDOR
IDR/USD rate with an ARIMAX(2)-GARCH(1,1) model using the Federal Funds Rate (FFR),
the Bank Indonesia policy rate (BI), and the Indonesian CPI as exogenous variables,
and benchmarks it against a driftless random walk, a random walk with drift, a nested
ARIMA-GARCH control, and an ARIMAX-ECM-GARCH extension, on both point and interval
forecasts under a leakage-free rolling-window protocol.

## Contents

| File | Description |
|------|-------------|
| `IDRUSD_ARIMAX_GARCH_clean.ipynb` | Main notebook — reproduces every table and figure, top to bottom. |
| `forecast_idrusd_arimax_garch.py` | Script export of the same analysis. |
| `merged_data_clean.csv` | Daily data: `date, fx, ffr, cpi, bi` (2 Apr 2018 – 27 Jul 2026). |
| `expected_output.txt` | Console output of a full run, for verification. |
| `requirements.txt` | Python dependencies. |

## Data

`merged_data_clean.csv` holds 2,004 daily trading-day observations. `fx` is the JISDOR
rate (Bank Indonesia); `ffr` is the Federal Funds Rate (FRED series DFEDTARU); `cpi` is
the Indonesian CPI reconstructed from the monthly inflation releases (BPS); `bi` is the
BI policy rate. The monthly CPI and the policy rates are aligned to the daily grid by
last-observation-carried-forward.

## How to run

```bash
pip install -r requirements.txt
```

Then open `IDRUSD_ARIMAX_GARCH_clean.ipynb` (Jupyter or Google Colab) and run all cells,
or run the script:

```bash
python forecast_idrusd_arimax_garch.py
```

Both read `merged_data_clean.csv` from the working directory. A full run takes a few
minutes (the interval section simulates 2,000 paths per origin).

## What it produces (maps to the article)

| Section | Output |
|---------|--------|
| 3 | Table I — ADF & Phillips-Perron unit-root tests |
| 4 | Johansen cointegration test (trace = 70.14, rank r = 1) |
| 5 | Table II — ARIMAX(2)-GARCH(1,1) estimates (representative window + cross-window); Figure 6 |
| 6-7 | Table III — point-forecast accuracy of the five models |
| 8 | Tables IV & V — Clark-West tests (one-sided, log-return space, HAC lag h-1) |
| 9 | 95% interval evaluation (coverage, width, interval score, Kupiec, Christoffersen); Figure 5 |
| 10 | Figures 1-4 |

## Method notes

- Model orders (AR(2), GARCH(1,1)) are frozen in advance; only coefficients are
  re-estimated on each 1,000-observation rolling window.
- Multi-step forecasts hold the exogenous variables at their last observed level, so
  their future first differences are zero — no future macro value is ever read.
- Point-forecast significance uses the Clark-West statistic for nested models, computed
  in log-return space with HAC (Newey-West) standard errors and lag order h-1.
- The GARCH(1,1) is symmetric (`GARCH(p=1, o=0, q=1)`).

## License

MIT (see `LICENSE`).
