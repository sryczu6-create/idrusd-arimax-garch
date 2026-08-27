# Leakage-free rolling-window evaluation of macroeconomic models for IDR/USD forecasting

Replication code for *"Evaluasi Rolling-Window Model Makroekonomi untuk Peramalan
Nilai Tukar Rupiah"* — ARIMAX(2)-GARCH(1,1) vs a nested ARIMA-GARCH control, a
random walk with and without drift, and an ARIMAX-ECM-GARCH extension, all under a
strictly leakage-free rolling-window protocol.

## Repository layout

```
forecast_idrusd_arimax_garch.py   # the whole analysis, one script
merged_data_clean.csv             # input data (date, fx, ffr, cpi, bi)
requirements.txt                  # Python dependencies
expected_output.txt               # reference console output (for comparison)
README.md
.gitignore
```

## What it reproduces

| Output | Section in the script |
|--------|-----------------------|
| Table 1 — ADF & Phillips-Perron unit-root tests | §3 |
| Table 2 — representative-window estimates + residual diagnostics | §4 |
| Table 2B — cross-window coefficient distribution + macro p-value figure | §4B |
| Johansen cointegration test | §6 |
| Table 3 — point-forecast accuracy (MAE / RMSE / MAPE) | §8 |
| Tables 4 & 5 — Clark-West MSPE-adjusted tests (log-return space) | §9 |
| Figures 1–4 | §10 |

Forecast accuracy is compared with the **Clark-West** MSPE-adjusted test — the
appropriate one-sided statistic for nested models — with HAC (Newey-West, lag = h−1)
standard errors for multi-step overlap. Tables 4 & 5 are reported on log-return
errors (retransformation-free); the same test on level errors is printed as a
robustness appendix.

## Anti-leakage design

- Model orders (AR(2), GARCH(1,1)) are frozen in advance; only the coefficients are
  re-estimated on each rolling window.
- Every window uses only observations up to the forecast origin.
- For h > 1, exogenous variables are held at their last known level at the origin, so
  their future first differences are zero — no future macro value is ever read.

## Data

`merged_data_clean.csv`, daily, 2 April 2018 – 27 July 2026 (2004 rows):

| Column | Description | Source |
|--------|-------------|--------|
| `date` | trading day | — |
| `fx`   | IDR/USD JISDOR rate | Bank Indonesia |
| `ffr`  | Federal Funds Rate | FRED, series `DFEDTARU` |
| `cpi`  | Indonesia CPI / IHK | BPS |
| `bi`   | BI policy rate | Bank Indonesia |

## Run

```bash
pip install -r requirements.txt
python forecast_idrusd_arimax_garch.py
```

All tables print to stdout; figures (`figure1_level.png` … `figure4_mae_reduction.png`,
`fig_macro_pvalues.png`) and `table_2b.csv` are written to the working directory.
Full run takes ~3 minutes (1003 rolling windows, three models re-fit per window).
A reference console log is in `expected_output.txt`.

Tested with numpy 2.4, pandas 3.0, scipy 1.16, statsmodels 0.14, arch 8.0.

## License

Not set yet — add a `LICENSE` file before publishing (the article is CC-BY-SA; MIT
or CC-BY-SA are common choices for accompanying code).
