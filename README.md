# Do Macroeconomic Variables Improve Rupiah Forecasts under Conditional Heteroskedasticity?

Code and data for the article of the same title by Sry Yulianti Lobo, Rahmat Syam, and Wahidah Sanusi
(Universitas Negeri Makassar), submitted to the *Journal of Applied Informatics and Computing (JAIC)*.

The study tests whether three macroeconomic variables (Federal Funds Rate, Bank Indonesia policy rate,
and Indonesian CPI) improve daily IDR/USD (JISDOR) forecasts from a linear ARIMA-GARCH model. The resulting
ARIMAX(2)-GARCH(1,1) is compared with a nested ARIMA-GARCH control, a random walk without drift, a random walk
with drift, and an ARIMAX-ECM-GARCH extension, on point and 95% interval forecasts, under a leakage-free
fixed-size rolling-window protocol.

## Contents

| File | Description |
|------|-------------|
| `IDRUSD_ARIMAX_GARCH.ipynb` | Main notebook with outputs. Reproduces every table and figure in the article, top to bottom. |
| `forecast_idrusd_arimax_garch.py` | Script export of the same notebook. |
| `merged_data_clean.csv` | Daily data: `date, fx, ffr, cpi, bi` (2 April 2018 to 27 July 2026, 2,004 observations). |
| `figures/` | Figures 1 to 6 of the article (600 dpi). |
| `requirements.txt` | Python dependencies with the exact versions used. |
| `LICENSE` | MIT license. |

## Data

`merged_data_clean.csv` holds 2,004 daily trading-day observations, which give 2,003 daily log returns.

| Column | Variable | Source |
|--------|----------|--------|
| `fx` | JISDOR IDR/USD rate | Bank Indonesia |
| `ffr` | Federal Funds Target Range, upper limit | FRED series DFEDTARU |
| `bi` | BI policy rate | Bank Indonesia |
| `cpi` | Indonesian CPI, reconstructed by chaining monthly month-to-month inflation | Statistics Indonesia (BPS) |

The JISDOR series is in trading-day time, so non-trading days are absent rather than interpolated.
The policy rates and the CPI are aligned to the daily grid by last-observation-carried-forward.
The data cutoff is 27 July 2026.

## How to run

Open `IDRUSD_ARIMAX_GARCH.ipynb` in Google Colab or Jupyter, upload `merged_data_clean.csv` to the working
directory, and run all cells. Alternatively:

```bash
pip install -r requirements.txt
python forecast_idrusd_arimax_garch.py
```

A full run takes a while on Colab, mainly because of the interval simulation (2,000 paths per origin)
and the six robustness specifications, each of which re-estimates both models on every rolling window.

## Notebook sections and article outputs

| Section | Output in the article |
|---------|-----------------------|
| 3 | Table I, ADF and Phillips-Perron unit-root tests |
| 4 | Johansen cointegration test, trace and maximum-eigenvalue (Section III.B) |
| 4a, 4b | ACF/PACF and model-order selection by AIC, BIC, and Ljung-Box (Section III.B) |
| 5.1 | Table II left block and residual diagnostics for both models (Section III.C) |
| 5.2, 5.3 | Table II right block, cross-window distribution over 1,004 windows |
| 6, 7 | Table III, point-forecast accuracy |
| 8 | Tables IV and V, Clark-West tests with RMSE differences |
| 8b | Giacomini-White tests and Holm-adjusted p-values (Section III.D) |
| 9 | Table VI, coverage, width, interval score, Kupiec and Christoffersen tests |
| 11 | Robustness checks with Student-t errors, windows of 750 and 1,250, EGARCH, and GJR-GARCH (Section III.F) |
| 11b | Subperiod robustness, tightening and post-tightening (Section III.F) |
| 12 | Figures 1 to 6 |

## Method notes

- Model orders (AR(2), GARCH(1,1)) are fixed before the rolling evaluation; only the coefficients are
  re-estimated on each 1,000-observation window.
- The first window spans 3 April 2018 to 19 May 2022, so the out-of-sample period runs from
  20 May 2022 to 27 July 2026 (1,003 one-day forecast origins, 944 at the quarterly horizon).
- At every horizon, including h = 1, the future differences of the exogenous variables are set to zero,
  so no future macroeconomic value is ever read.
- Clark-West and Giacomini-White tests are computed in log-return space, with HAC lag h - 1 and
  n^(1/3) respectively, and p-values are also Holm-adjusted across the five horizons.
- Interval simulations use a fixed random seed of 42.

## Software versions

Python 3.13, arch 8.0.0, statsmodels 0.15.0, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3.

## License

MIT (see `LICENSE`).
