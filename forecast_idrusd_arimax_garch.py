# -*- coding: utf-8 -*-
# =====================================================================
#  ARIMAX(2)-GARCH(1,1) vs nested ARIMA-GARCH vs Random Walk (+ drift)
#  vs ARIMAX-ECM-GARCH
#  Leakage-free rolling-window forecast evaluation for IDR/USD (JISDOR)
# ---------------------------------------------------------------------
#  Reproduces: Table 1 (unit root), Table 2 + 2B (estimates),
#  Table 3 (accuracy), Table 4 & 5 (Clark-West), Johansen test,
#  and Figures 1-4.
#
#  Accuracy is compared with the Clark-West MSPE-adjusted test, which is
#  the appropriate one-sided statistic for nested models. (An earlier
#  draft used Diebold-Mariano-HLN; that has been removed.)
#
#  Key anti-leakage design:
#  * Model orders (AR(2), GARCH(1,1)) are FROZEN in advance; only the
#    coefficients are re-estimated on each window.
#  * Each window uses ONLY observations up to the forecast origin `o`.
#  * For multi-step (h > 1) forecasts the exogenous variables are held at
#    their last known level at the origin, so their future first
#    differences are ZERO. No future macro value (FFR/BI/CPI) is ever
#    read -- exactly the information a real-time forecaster has.
#
#  Requirements: numpy, pandas, scipy, statsmodels, arch, matplotlib
#  (see requirements.txt).  Input: merged_data_clean.csv with columns
#  date, fx, ffr, bi, cpi.
# =====================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

from scipy import stats
from arch.univariate import ARX, GARCH, Normal
from arch.unitroot import ADF, PhillipsPerron
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

# --------------------------- configuration ---------------------------
DATA_FILE = "merged_data_clean.csv"      # columns: date, fx, ffr, bi, cpi
WINDOW    = 1000                          # rolling window length (return obs)
H         = [1, 5, 10, 20, 60]           # forecast horizons (trading days)
ALPHA     = 0.05
np.random.seed(42)                       # reproducibility


# =====================================================================
# 1. Load data
# =====================================================================
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()


# =====================================================================
# 2. Preprocessing: stationary transforms
#    fx        -> log-return (in %), I(0)
#    exogenous -> first difference (each is I(1))
# =====================================================================
df["logret"] = np.log(df["fx"]).diff() * 100.0
for c in ["ffr", "bi", "cpi"]:
    df["d_" + c] = df[c].diff()

data = df.dropna().copy()
y    = data["logret"].to_numpy()
Xz   = data[["d_ffr", "d_bi", "d_cpi"]].to_numpy()
lvl  = data["fx"].to_numpy()
n    = len(data)

print("=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"Aligned observations : {n}")
print(f"Sample period        : {data.index[0].date()}  to  {data.index[-1].date()}")
print(f"Max fx               : {lvl.max():.0f}  on {data.index[int(np.argmax(lvl))].date()}")
print(f"Min fx               : {lvl.min():.0f}  on {data.index[int(np.argmin(lvl))].date()}")


# =====================================================================
# 3. Stationarity tests (Table 1): ADF & PP on level and first difference
# =====================================================================
print("\n" + "=" * 70)
print("TABLE 1  Unit-root tests (p-values)")
print("=" * 70)
print(f"{'Series':<18}{'ADF lvl':>9}{'PP lvl':>9}{'ADF d':>9}{'PP d':>9}{'  Order':>8}")

def pp_pair(level_series, name):
    lv = level_series.dropna().to_numpy()
    dv = np.diff(lv)
    adf_l = ADF(lv).pvalue;  pp_l = PhillipsPerron(lv).pvalue
    adf_d = ADF(dv).pvalue;  pp_d = PhillipsPerron(dv).pvalue
    order = "I(1)" if (adf_l > 0.05 and adf_d < 0.05) else "check"
    print(f"{name:<18}{adf_l:>9.4f}{pp_l:>9.4f}{adf_d:>9.4f}{pp_d:>9.4f}{order:>8}")

pp_pair(np.log(df["fx"]), "log(fx)")
for c, nm in [("ffr", "Federal Funds"), ("bi", "BI Rate"), ("cpi", "CPI")]:
    pp_pair(df[c], nm)


# =====================================================================
# 4. Representative-window estimation (Table 2) + residual diagnostics
#    Uses the final full window as the representative window.
# =====================================================================
rep  = slice(n - WINDOW, n)                      # last WINDOW observations
mrep = ARX(y[rep], lags=[1, 2], x=Xz[rep], rescale=False)
mrep.volatility = GARCH(p=1, o=0, q=1)
rres = mrep.fit(disp="off")

print("\n" + "=" * 70)
print("TABLE 2  ARIMAX(2)-GARCH(1,1) estimates (representative window)")
print("=" * 70)
print(rres.summary())

zr = np.asarray(rres.std_resid)
zr = zr[~np.isnan(zr)]
lb        = acorr_ljungbox(zr, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
arch_lm_p = het_arch(zr, nlags=12)[1]
jb_p      = stats.jarque_bera(zr)[1]
print("\nResidual diagnostics (standardized residuals):")
print(f"  Ljung-Box(20) p = {lb:.3f}   ARCH-LM p = {arch_lm_p:.3f}   Jarque-Bera p = {jb_p:.3g}")


# =====================================================================
# 4B. Cross-window coefficient distribution (Table 2B)
#     + Figure: per-window p-values of the macro regressors
# =====================================================================
w = pd.DataFrame(index=df.index)
w["logret"] = 100.0 * np.log(df["fx"]).diff()    # daily log-return in percent
w["dFFR"]   = df["ffr"].diff()
w["dBI"]    = df["bi"].diff()
w["dCPI"]   = df["cpi"].diff()
w = w.dropna()

EXOG = ["dFFR", "dBI", "dCPI"]
y_all, x_all = w["logret"], w[EXOG]
n_windows = len(w) - WINDOW + 1
print(f"\n{len(w)} obs -> {n_windows} rolling windows")

rows, failed = [], 0
for i in range(n_windows):
    sl = slice(i, i + WINDOW)
    m = ARX(y_all.iloc[sl], x=x_all.iloc[sl], lags=[1, 2], constant=True,
            volatility=GARCH(1, 1), distribution=Normal())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = m.fit(disp="off", cov_type="robust")   # QMLE / Bollerslev-Wooldridge SE
        except Exception:
            failed += 1; continue
    for name in r.params.index:
        rows.append({"param": name, "coef": float(r.params[name]),
                     "pval": float(r.pvalues[name])})
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{n_windows}")
if failed:
    print(f"WARNING: {failed} windows skipped (no convergence)")

long_df = pd.DataFrame(rows)

summary = (long_df.groupby("param")
    .agg(mean_coef=("coef", "mean"), median_coef=("coef", "median"),
         sd_coef=("coef", "std"),
         pct_positive=("coef", lambda s: 100 * (s > 0).mean()),
         mean_pval=("pval", "mean"),
         pct_sig=("pval", lambda s: 100 * (s < ALPHA).mean())))
order   = ["Const", "logret[1]", "logret[2]", "dFFR", "dBI", "dCPI",
           "omega", "alpha[1]", "beta[1]"]
summary = summary.reindex([p for p in order if p in summary.index]).round(4)
print("\n=== Table 2B: cross-window coefficient summary ===")
print(summary.to_string())
summary.to_csv("table_2b.csv")

# NB: use a dedicated name (box_pvals) so we do NOT overwrite `data`.
box_pvals = [long_df.loc[long_df["param"] == c, "pval"].values for c in EXOG]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.boxplot(box_pvals, labels=EXOG, showfliers=False)
ax.axhline(ALPHA, ls="--", lw=1, label=f"p = {ALPHA}")
ax.set_ylabel("p-value across rolling windows")
ax.set_title("Per-window significance of macro regressors")
ax.legend()
fig.tight_layout(); fig.savefig("fig_macro_pvalues.png", dpi=200); plt.close(fig)

macro_sig = summary.loc[summary.index.isin(EXOG), "pct_sig"]
print(f"\nMacro regressors significant at 5% in "
      f"{macro_sig.min():.1f}-{macro_sig.max():.1f}% of windows (chance ~5%).")


# =====================================================================
# 5. Rolling-window, leakage-free forecast errors
#    Builds a single `err` dict shared by every table and figure:
#      err['rw'], err['arima'], err['arimax'], err['rw_drift'],
#      err['arimax_ecm']
#    Also stores actuals `act` (for Figure 3).
# =====================================================================
err = {m: {h: [] for h in H} for m in ["rw", "arima", "arimax"]}
act = {h: [] for h in H}
exog_future = np.zeros(3)   # future FFR/BI/CPI diffs are unknown at origin -> 0

for o in range(WINDOW - 1, n - 1):              # o = index of last in-sample obs
    tr = slice(o - WINDOW + 1, o + 1)           # WINDOW obs, all indices <= o
    ytr, Xtr = y[tr], Xz[tr]

    # ARIMAX(2)-GARCH(1,1)  (full model with exogenous regressors)
    mx = ARX(ytr, lags=[1, 2], x=Xtr, rescale=False)
    mx.volatility = GARCH(p=1, o=0, q=1)
    px = mx.fit(disp="off").params.to_numpy()
    cX, a1X, a2X, bX = px[0], px[1], px[2], px[3:6]

    # ARIMA(2)-GARCH(1,1)  (nested control: same dynamics, no exogenous)
    m0 = ARX(ytr, lags=[1, 2], rescale=False)
    m0.volatility = GARCH(p=1, o=0, q=1)
    p0 = m0.fit(disp="off").params.to_numpy()
    c0, a10, a20 = p0[0], p0[1], p0[2]

    for h in H:
        if o + h > n - 1:
            continue
        # ---- ARIMAX iterated h-step forecast (exog diff = 0, no leakage) ----
        l1, l2, s = y[o], y[o - 1], 0.0
        for _ in range(h):
            rh = cX + a1X * l1 + a2X * l2 + bX @ exog_future
            s += rh; l2, l1 = l1, rh
        lx = lvl[o] * np.exp(s / 100.0)
        # ---- ARIMA iterated h-step forecast ----
        l1, l2, s = y[o], y[o - 1], 0.0
        for _ in range(h):
            rh = c0 + a10 * l1 + a20 * l2
            s += rh; l2, l1 = l1, rh
        l0 = lvl[o] * np.exp(s / 100.0)
        # ---- Random walk (no-change) forecast ----
        lrw = lvl[o]

        a = lvl[o + h]
        act[h].append(a)
        err["arimax"][h].append(a - lx)
        err["arima"][h].append(a - l0)
        err["rw"][h].append(a - lrw)


# =====================================================================
# 6. Johansen cointegration test (motivates the ECM extension)
# =====================================================================
joh = pd.DataFrame({
    "logX": np.log(df["fx"]),
    "ffr":  df["ffr"],
    "bi":   df["bi"],
    "cpi":  df["cpi"],
}).dropna()

p_lag = max(int(VAR(joh).select_order(maxlags=10).aic), 1)   # AIC lag -> k_ar_diff = p-1
res   = coint_johansen(joh, det_order=0, k_ar_diff=p_lag - 1)

def report(name, stat, crit):
    print("\n" + name)
    print(f"{'r':<6}{'stat':>10}{'cv90%':>10}{'cv95%':>10}{'cv99%':>10}   decision (5%)")
    for i, (s, c) in enumerate(zip(stat, crit)):
        print(f"{i:<6}{s:>10.2f}{c[0]:>10.2f}{c[1]:>10.2f}{c[2]:>10.2f}   "
              f"{'reject H0' if s > c[1] else 'fail to reject'}")

print("\n" + "=" * 70)
print("JOHANSEN COINTEGRATION TEST")
print("=" * 70)
print(f"Observations : {len(joh)}")
print(f"Variables    : {list(joh.columns)}")
print(f"Lag (AIC)    : {p_lag}   (k_ar_diff = {p_lag - 1})")
report("TRACE TEST", res.lr1, res.cvt)
report("MAX-EIGENVALUE TEST", res.lr2, res.cvm)

r_rank = 0
for s, c in zip(res.lr1, res.cvt):
    if s > c[1]:
        r_rank += 1
    else:
        break
print(f"\nCointegration rank (trace, 5%): r = {r_rank}")


# =====================================================================
# 7. ARIMAX-ECM-GARCH errors  and  Random-walk-with-drift errors
#    ARIMAX-ECM = model 4 with one extra regressor z_{t-1}:
#      dlogX_t = c + phi1 dlogX_{t-1} + phi2 dlogX_{t-2}
#                + theta' dW_t + lambda z_{t-1} + eps_t ,  eps ~ GARCH(1,1)
#      z_{t-1} = logX_{t-1} - (a + b' W_{t-1})   (deviation from equilibrium)
#    lambda = 0 collapses it to model 4, so the two are directly comparable.
#    The cointegrating vector is re-estimated on each window's data only.
# =====================================================================
price = lvl.astype(float)                                  # FX level, aligned to y
Wlvl  = data[["ffr", "cpi", "bi"]].to_numpy(dtype=float)   # macro levels, aligned to y
logp  = np.log(price)
assert len(price) == len(y) == len(Xz) == len(Wlvl) == n, "length mismatch"

def fit_ec_term(logp_win, Wlvl_win):
    # Engle-Granger long-run regression on IN-WINDOW LEVELS only (leakage-free):
    #   logX = a + b'W + z   ->   z = equilibrium error (the EC term)
    D = np.column_stack([np.ones(len(logp_win)), Wlvl_win])
    coef, *_ = np.linalg.lstsq(D, logp_win, rcond=None)
    z = logp_win - D @ coef
    return coef[0], coef[1:], z

err["arimax_ecm"] = {h: [] for h in H}
for o in range(WINDOW - 1, n - 1):
    tr  = slice(o - WINDOW + 1, o + 1)
    ytr = y[tr]

    a_ec, b_ec, z_win = fit_ec_term(logp[tr], Wlvl[tr])     # equilibrium on THIS window
    z_lag = np.empty_like(z_win)                            # row t carries z_{t-1}
    z_lag[0]  = 0.0
    z_lag[1:] = z_win[:-1]

    fit_ok = True
    try:
        Xe = np.column_stack([Xz[tr], z_lag])               # 3 diffs + z_{t-1}
        me = ARX(ytr, lags=[1, 2], x=Xe, rescale=False)
        me.volatility = GARCH(p=1, o=0, q=1)
        pe = me.fit(disp="off").params.to_numpy()
        cE, a1E, a2E = pe[0], pe[1], pe[2]
        lamE = pe[6]                                        # lambda on z_{t-1}
    except Exception:
        fit_ok = False                                      # rare non-convergence

    for h in H:
        if o + h > n - 1:
            continue
        if not fit_ok:
            err["arimax_ecm"][h].append(price[o] - price[o + h])
            continue
        r1, r2 = y[o], y[o - 1]
        logp_k = logp[o]
        Weq    = a_ec + b_ec @ Wlvl[o]                      # frozen (dW_future = 0)
        for k in range(1, h + 1):
            z_prev = logp_k - Weq
            yhat   = cE + a1E * r1 + a2E * r2 + lamE * z_prev
            logp_k += yhat / 100.0
            r2, r1 = r1, yhat
        err["arimax_ecm"][h].append(np.exp(logp_k) - price[o + h])

err["rw_drift"] = {h: [] for h in H}
for o in range(WINDOW - 1, n - 1):
    tr    = slice(o - WINDOW + 1, o + 1)
    drift = np.mean(y[tr]) / 100.0                          # mean log-return, info <= o
    for h in H:
        if o + h > n - 1:
            continue
        xhat = price[o] * np.exp(h * drift)                 # RW-with-drift level forecast
        err["rw_drift"][h].append(xhat - price[o + h])


# =====================================================================
# 8. Table 3: point-forecast accuracy (MAE & RMSE in IDR, MAPE in %)
# =====================================================================
MODELS  = [m for m in ["rw", "rw_drift", "arima", "arimax", "arimax_ecm"] if m in err]
origins = range(WINDOW - 1, n - 1)
actuals = {h: np.array([price[o + h] for o in origins if o + h <= n - 1], float) for h in H}

def metrics(e, a):
    e = np.asarray(e, float)
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2)), 100 * np.mean(np.abs(e) / a)

print("\n" + "=" * 70)
print("TABLE 3  Point-forecast accuracy")
print("=" * 70)
for h in H:
    a = actuals[h]
    print(f"\nHorizon h = {h}")
    for m in MODELS:
        mae, rmse, mape = metrics(err[m][h], a)
        print(f"  {m:<12} MAE={mae:8.2f}   RMSE={rmse:8.2f}   MAPE={mape:7.4f}")
print("\nN forecasts per horizon:", {h: len(actuals[h]) for h in H})


# =====================================================================
# 9. Clark-West (MSPE-adjusted) test for nested forecast comparison
#    One-sided; H1: the larger (nesting) model is more accurate.
#    HAC (Newey-West) SE with lag = h-1 handles multi-step overlap.
#    Reported on LOG-RETURN errors (retransformation-free), as in the
#    paper; also on LEVEL errors as a robustness appendix.
# =====================================================================
def clark_west(e_small, e_large, h):
    e_small = np.asarray(e_small, float)
    e_large = np.asarray(e_large, float)
    f = e_small ** 2 - e_large ** 2 + (e_small - e_large) ** 2   # CW loss differential
    f = f[~np.isnan(f)]
    t = float(sm.OLS(f, np.ones((f.size, 1)))
              .fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)}).tvalues[0])
    return {"t": t,
            "p": float(1.0 - stats.norm.cdf(t)),             # one-sided
            "rmse_small": float(np.nanmean(e_small ** 2)) ** 0.5,
            "rmse_large": float(np.nanmean(e_large ** 2)) ** 0.5,
            "n": int(f.size)}

def detect_keys(err_dict):
    """Map raw err keys -> canonical labels by name pattern."""
    m = {}
    for k in err_dict:
        s = k.lower()
        if   "ecm"    in s:                m["ARIMAX-ECM-GARCH"] = k
        elif "arimax" in s:                m["ARIMAX-GARCH"]     = k
        elif "arima"  in s:                m["ARIMA-GARCH"]      = k
        elif "drift"  in s:                m["RW-drift"]         = k
        elif "rw" in s or "random" in s:   m["RW"]               = k
    return m

def actuals_by_h(price_arr, window, horizons):
    price_arr = np.asarray(price_arr, float); N = len(price_arr)
    return {h: np.array([price_arr[o + h] for o in range(window - 1, N - 1)
                         if o + h <= N - 1]) for h in horizons}

def to_logret_errors(err_dict, price_arr, window, horizons):
    """Exact identity: e_ret = log(forecast/actual) = log(1 + e_level/actual)."""
    act_h = actuals_by_h(price_arr, window, horizons)
    out = {}
    for mdl in err_dict:
        out[mdl] = {}
        for h in horizons:
            e = np.asarray(err_dict[mdl][h], float); a = act_h[h]
            assert len(e) == len(a), f"len mismatch {mdl} h={h}: {len(e)} vs {len(a)}"
            out[mdl][h] = np.log1p(e / a)
    return out

def cw_report(err_dict, pairs, horizons, title):
    """Print CW p-value, direction, and signed %RMSE gap for each (big, small) pair."""
    km = detect_keys(err_dict)
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'big model':<18}{'vs':<14}{'h':>4}{'N':>6}{'t_CW':>8}"
          f"{'p':>8}{'RMSE gap%':>11}{'  better?':>10}")
    for big, small in pairs:
        for h in horizons:
            r = clark_west(err_dict[km[small]][h], err_dict[km[big]][h], h)
            gap = 100 * (r["rmse_small"] - r["rmse_large"]) / r["rmse_small"]
            better = "yes" if r["rmse_large"] < r["rmse_small"] else "NO"
            print(f"{big:<18}{small:<14}{h:>4}{r['n']:>6}{r['t']:>8.2f}"
                  f"{r['p']:>8.3f}{gap:>+11.2f}{better:>10}")

# Table 4 pairs: ARIMAX-GARCH vs its nested benchmarks
# Table 5 pairs: ARIMAX-ECM-GARCH vs its nested benchmarks
PAIRS = [
    ("ARIMAX-GARCH",     "RW"), ("ARIMAX-GARCH",     "RW-drift"), ("ARIMAX-GARCH", "ARIMA-GARCH"),
    ("ARIMAX-ECM-GARCH", "RW"), ("ARIMAX-ECM-GARCH", "RW-drift"),
    ("ARIMAX-ECM-GARCH", "ARIMA-GARCH"), ("ARIMAX-ECM-GARCH", "ARIMAX-GARCH"),
]

err_ret = to_logret_errors(err, price, WINDOW, H)
cw_report(err_ret, PAIRS, H, "TABLE 4 & 5  Clark-West on LOG-RETURN errors (primary, as reported)")
cw_report(err,     PAIRS, H, "ROBUSTNESS   Clark-West on LEVEL errors")


# =====================================================================
# 10. Figures 1-4
# =====================================================================
dates = data.index

# ---- Figure 1: level of the exchange rate ----
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(df.index, df["fx"], color="blue", lw=1.0)
ax.set_title("IDR/USD Exchange Rate (JISDOR), 2018-2026")
ax.set_xlabel("Date"); ax.set_ylabel("IDR per USD")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.savefig("figure1_level.png", dpi=200); plt.close(fig)

# ---- Figure 2: log-returns and squared returns (volatility clustering) ----
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
a1.plot(dates, y, color="red", lw=0.6)
a1.axhline(0, color="0.5", lw=0.6)
a1.set_title("Daily log-returns of the IDR/USD rate (%)")
a1.set_ylabel("Return (%)")
a2.plot(dates, y ** 2, color="purple", lw=0.6)
a2.set_title("Squared daily returns (volatility clustering)")
a2.set_ylabel("Squared return (%$^2$)"); a2.set_xlabel("Date")
a2.xaxis.set_major_locator(mdates.YearLocator())
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.savefig("figure2_returns.png", dpi=200); plt.close(fig)

# ---- Figure 3: last 250 one-day-ahead forecasts, actual vs models ----
f3_idx    = list(range(WINDOW, n))
f3_actual = np.array(act[1])
f3_arimax = np.array(act[1]) - np.array(err["arimax"][1])
f3_rw     = np.array(act[1]) - np.array(err["rw"][1])
k   = min(250, len(f3_idx))
sel = slice(len(f3_idx) - k, len(f3_idx))
d3  = dates[np.array(f3_idx)[sel]]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(d3, f3_actual[sel], color="blue",  lw=1.4, label="Actual")
ax.plot(d3, f3_arimax[sel], color="green", lw=1.0, label="ARIMAX-GARCH")
ax.plot(d3, f3_rw[sel],     color="orange", lw=1.0, ls="--", label="Random walk")
ax.set_title(f"One-day-horizon (h=1) rolling forecasts: actual vs ARIMAX-GARCH vs random walk (last {k} days)")
ax.set_xlabel("Date"); ax.set_ylabel("IDR/USD")
ax.legend(frameon=False, loc="best")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.savefig("figure3_forecasts.png", dpi=200); plt.close(fig)

# ---- Figure 4: MAE reduction of ARIMAX-GARCH vs horizon (computed, not hardcoded) ----
red_rw, red_rwd, red_ar = [], [], []
for h in H:
    mae_x   = np.mean(np.abs(err["arimax"][h]))
    mae_r   = np.mean(np.abs(err["rw"][h]))
    mae_rwd = np.mean(np.abs(err["rw_drift"][h]))
    mae_a   = np.mean(np.abs(err["arima"][h]))
    red_rw.append(100 * (mae_r - mae_x) / mae_r)
    red_rwd.append(100 * (mae_rwd - mae_x) / mae_rwd)
    red_ar.append(100 * (mae_a - mae_x) / mae_a)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.axhline(0, color="0.5", lw=0.8)
ax.plot(H, red_rw,  "-o", color="#1f4e79", label="ARIMAX-GARCH vs driftless random walk")
ax.plot(H, red_rwd, "-s", color="#e07b00", label="ARIMAX-GARCH vs random walk with drift")
ax.plot(H, red_ar,  "-^", color="#4c9a2a", label="ARIMAX-GARCH vs ARIMA-GARCH (macro contribution)")
for x, v in zip(H, red_rw):
    ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9, color="#1f4e79")
ax.set_title("MAE reduction of ARIMAX-GARCH vs forecast horizon")
ax.set_xlabel("Forecast horizon (trading days)"); ax.set_ylabel("MAE reduction (%)")
ax.set_xticks(H); ax.legend(frameon=False, loc="best")
ax.grid(True, alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
fig.savefig("figure4_mae_reduction.png", dpi=300, bbox_inches="tight"); plt.close(fig)

print("\nFigures saved: figure1_level.png, figure2_returns.png, "
      "figure3_forecasts.png, figure4_mae_reduction.png, fig_macro_pvalues.png")
print("vs driftless RW :", [round(float(v), 1) for v in red_rw])
print("vs RW-with-drift:", [round(float(v), 1) for v in red_rwd])
print("vs ARIMA (macro):", [round(float(v), 1) for v in red_ar])
