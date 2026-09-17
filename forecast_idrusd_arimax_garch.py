import warnings; warnings.filterwarnings("ignore")
!pip install -q arch          # pin your version for exact reproducibility, e.g. arch==8.0.0
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from arch.univariate import ARX, GARCH
from arch.unitroot import ADF, PhillipsPerron
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

np.random.seed(42)
rng = np.random.default_rng(42)   # for the interval simulations

DATA_FILE = "merged_data_clean.csv"          # columns: date, fx, ffr, cpi, bi
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.set_index("date").sort_index()

df["logret"] = np.log(df["fx"]).diff() * 100.0
for c in ["ffr", "bi", "cpi"]:
    df["d_" + c] = df[c].diff()

data = df.dropna().copy()
y    = data["logret"].to_numpy()
Xz   = data[["d_ffr", "d_bi", "d_cpi"]].to_numpy()
lvl  = data["fx"].to_numpy()
logp = np.log(lvl)                                   # for the ECM model
Wlvl = data[["ffr", "cpi", "bi"]].to_numpy(float)    # macro levels for cointegration
n    = len(data)

print("=" * 70); print("DATA SUMMARY"); print("=" * 70)
print(f"Aligned observations : {n}")
print(f"Sample period        : {data.index[0].date()}  to  {data.index[-1].date()}")
print(f"Max fx               : {lvl.max():.0f}  on {data.index[int(np.argmax(lvl))].date()}")
print(f"Min fx               : {lvl.min():.0f}  on {data.index[int(np.argmin(lvl))].date()}")

print("=" * 70); print("TABLE I  Unit-root tests (p-values)"); print("=" * 70)
print(f"{'Series':<18}{'ADF lvl':>9}{'PP lvl':>9}{'ADF d':>9}{'PP d':>9}{'  Order':>8}")

def unit_root_row(level_series, name):
    lv = level_series.dropna().to_numpy(); dv = np.diff(lv)
    adf_l = ADF(lv).pvalue; pp_l = PhillipsPerron(lv).pvalue
    adf_d = ADF(dv).pvalue; pp_d = PhillipsPerron(dv).pvalue
    order = "I(1)" if (adf_l > 0.05 and adf_d < 0.05) else "check"
    print(f"{name:<18}{adf_l:>9.4f}{pp_l:>9.4f}{adf_d:>9.4f}{pp_d:>9.4f}{order:>8}")

unit_root_row(np.log(df["fx"]), "log(fx)")
for c, nm in [("ffr", "Federal Funds"), ("bi", "BI Rate"), ("cpi", "CPI")]:
    unit_root_row(df[c], nm)

joh = pd.DataFrame({"logX": np.log(df["fx"]), "ffr": df["ffr"],
                    "bi": df["bi"], "cpi": df["cpi"]}).dropna()
p = max(int(VAR(joh).select_order(maxlags=10).aic), 1)
res = coint_johansen(joh, det_order=0, k_ar_diff=p - 1)

print("=" * 70); print("JOHANSEN COINTEGRATION TEST"); print("=" * 70)
print(f"Observations : {len(joh)}    Lag (AIC) : {p}  (k_ar_diff = {p-1})")
print(f"\n{'r':<4}{'trace':>10}{'cv95':>10}   decision (5%)")
for i, (s, c) in enumerate(zip(res.lr1, res.cvt[:, 1])):
    print(f"{i:<4}{s:>10.2f}{c:>10.2f}   {'reject H0' if s > c else 'fail to reject'}")
r = 0
for s, c in zip(res.lr1, res.cvt[:, 1]):
    if s > c: r += 1
    else: break
print(f"\nCointegration rank (trace, 5%): r = {r}")

WINDOW = 1000
rep = slice(n - WINDOW, n)
mrep = ARX(y[rep], lags=[1, 2], x=Xz[rep], rescale=False)
mrep.volatility = GARCH(p=1, o=0, q=1)
rres = mrep.fit(disp="off")

print("=" * 70); print("Representative window (last 1000 obs) - full estimates"); print("=" * 70)
print(rres.summary())

zr = np.asarray(rres.std_resid); zr = zr[~np.isnan(zr)]
lb = acorr_ljungbox(zr, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
arch_lm_p = het_arch(zr, nlags=12)[1]; jb_p = stats.jarque_bera(zr)[1]
print("\nResidual diagnostics (standardized residuals):")
print(f"  Ljung-Box(20) p = {lb:.3f}    ARCH-LM p = {arch_lm_p:.3f}    Jarque-Bera p = {jb_p:.3g}")

ALPHA = 0.05
n_windows = n - WINDOW + 1
print(f"{n} obs -> {n_windows} rolling windows")

rows, failed = [], 0
for i in range(n_windows):
    sl = slice(i, i + WINDOW)
    m = ARX(y[sl], lags=[1, 2], x=Xz[sl])          # true GARCH(1,1): o=0
    m.volatility = GARCH(p=1, o=0, q=1)
    try:
        r = m.fit(disp="off", cov_type="robust")
    except Exception:
        failed += 1; continue
    for name in r.params.index:
        rows.append({"param": name, "coef": float(r.params[name]), "pval": float(r.pvalues[name])})
    if (i + 1) % 100 == 0: print(f"  {i + 1}/{n_windows}")
if failed: print(f"WARNING: {failed} windows skipped (no convergence)")

long_df = pd.DataFrame(rows)
summary = (long_df.groupby("param")
           .agg(mean_coef=("coef", "mean"), sd_coef=("coef", "std"),
                pct_positive=("coef", lambda s: 100 * (s > 0).mean()),
                pct_sig=("pval", lambda s: 100 * (s < ALPHA).mean())))

# Figure 6: distribution of the macro-regressor p-values across rolling windows
EXOG, EXOG_LABELS = ["x0", "x1", "x2"], ["ΔFFR", "ΔBI", "ΔCPI"]
box_data = [long_df.loc[long_df["param"] == c, "pval"].values for c in EXOG]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.boxplot(box_data, labels=EXOG_LABELS, showfliers=False)
ax.axhline(ALPHA, ls="--", lw=1, label=f"p = {ALPHA}")
ax.set_ylabel("p-value across rolling windows")
ax.set_title("Distribution of the macroeconomic regressor p-values across rolling windows")
ax.legend(); fig.tight_layout(); fig.savefig("figure6.png", dpi=200); plt.show()
macro_sig = summary.loc[EXOG, "pct_sig"]
print(f"Macro regressors significant at 5% in "
      f"{macro_sig.min():.1f}-{macro_sig.max():.1f}% of windows (chance ~5%).")

LABELS = {"Const": "Constant", "y[1]": "logret(t-1)", "y[2]": "logret(t-2)",
          "x0": "ΔFFR", "x1": "ΔBI", "x2": "ΔCPI",
          "omega": "ω", "alpha[1]": "α1", "beta[1]": "β1"}
MEAN_EQ = ["Const", "y[1]", "y[2]", "x0", "x1", "x2"]; VAR_EQ = ["omega", "alpha[1]", "beta[1]"]
def fmt_p(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
def make_row(name):
    return {"Parameter": LABELS[name],
            "Coef.": round(float(rres.params[name]), 4), "Std. err.": round(float(rres.std_err[name]), 4),
            "p-value": fmt_p(float(rres.pvalues[name])),
            "Mean": round(float(summary.loc[name, "mean_coef"]), 4), "SD": round(float(summary.loc[name, "sd_coef"]), 4),
            "% coef>0": round(float(summary.loc[name, "pct_positive"]), 1),
            "% windows p<0.05": round(float(summary.loc[name, "pct_sig"]), 1)}
table2 = pd.DataFrame([make_row(k) for k in MEAN_EQ + VAR_EQ]).set_index("Parameter")
print("=" * 90); print("TABLE II  ARIMAX(2)-GARCH(1,1) ESTIMATES, REPRESENTATIVE WINDOW AND CROSS-WINDOW"); print("=" * 90)
print("Mean Equation"); print(table2.loc[[LABELS[k] for k in MEAN_EQ]].to_string())
print("\nVariance Equation"); print(table2.loc[[LABELS[k] for k in VAR_EQ]].to_string())
print(f"\nNote: Returns in percent; N = {int(rres.nobs)}; log-likelihood = {rres.loglikelihood:.2f}; "
      f"AIC = {rres.aic:.2f}; BIC = {rres.bic:.2f}.")
table2.to_csv("table2.csv")

H = [1, 5, 10, 20, 60]
err = {m: {h: [] for h in H} for m in ["arimax", "arima", "rw"]}
act = {h: [] for h in H}
exog_future = np.zeros(3)               # future macro diffs are unknown at the origin -> 0

for o in range(WINDOW - 1, n - 1):      # o = index of last in-sample observation
    tr = slice(o - WINDOW + 1, o + 1)
    ytr, Xtr = y[tr], Xz[tr]
    mx = ARX(ytr, lags=[1, 2], x=Xtr, rescale=False); mx.volatility = GARCH(p=1, o=0, q=1)
    px = mx.fit(disp="off").params.to_numpy(); cX, a1X, a2X, bX = px[0], px[1], px[2], px[3:6]
    m0 = ARX(ytr, lags=[1, 2], rescale=False); m0.volatility = GARCH(p=1, o=0, q=1)
    p0 = m0.fit(disp="off").params.to_numpy(); c0, a10, a20 = p0[0], p0[1], p0[2]
    for h in H:
        if o + h > n - 1: continue
        l1, l2, s = y[o], y[o - 1], 0.0
        for _ in range(h):
            rh = cX + a1X*l1 + a2X*l2 + bX @ exog_future; s += rh; l2, l1 = l1, rh
        lx = lvl[o] * np.exp(s / 100.0)
        l1, l2, s = y[o], y[o - 1], 0.0
        for _ in range(h):
            rh = c0 + a10*l1 + a20*l2; s += rh; l2, l1 = l1, rh
        l0 = lvl[o] * np.exp(s / 100.0)
        a = lvl[o + h]; act[h].append(a)
        err["arimax"][h].append(a - lx); err["arima"][h].append(a - l0); err["rw"][h].append(a - lvl[o])
    if (o - WINDOW + 2) % 200 == 0: print(f"  {o - WINDOW + 2}/{n - WINDOW}")
print("base loop done")

err["rw_drift"] = {h: [] for h in H}
for o in range(WINDOW - 1, n - 1):
    tr = slice(o - WINDOW + 1, o + 1)
    drift = np.mean(y[tr]) / 100.0                 # mean log-return in the window
    for h in H:
        if o + h > n - 1: continue
        err["rw_drift"][h].append(lvl[o] * np.exp(h * drift) - lvl[o + h])
print("rw_drift done")

def fit_ec_term(logp_win, Wlvl_win):
    D = np.column_stack([np.ones(len(logp_win)), Wlvl_win])
    coef, *_ = np.linalg.lstsq(D, logp_win, rcond=None)
    return coef[0], coef[1:], logp_win - D @ coef

err["arimax_ecm"] = {h: [] for h in H}
for o in range(WINDOW - 1, n - 1):
    tr = slice(o - WINDOW + 1, o + 1); ytr = y[tr]
    a_ec, b_ec, zw = fit_ec_term(logp[tr], Wlvl[tr])
    zlag = np.empty_like(zw); zlag[0] = 0.0; zlag[1:] = zw[:-1]
    ok = True
    try:
        me = ARX(ytr, lags=[1, 2], x=np.column_stack([Xz[tr], zlag]), rescale=False)
        me.volatility = GARCH(p=1, o=0, q=1)
        pe = me.fit(disp="off").params.to_numpy()
        cE, a1E, a2E, lamE = pe[0], pe[1], pe[2], pe[6]
    except Exception:
        ok = False
    for h in H:
        if o + h > n - 1: continue
        if not ok:
            err["arimax_ecm"][h].append(lvl[o] - lvl[o + h]); continue
        r1, r2 = y[o], y[o - 1]; logp_k = logp[o]; Weq = a_ec + b_ec @ Wlvl[o]
        for k in range(1, h + 1):
            yhat = cE + a1E*r1 + a2E*r2 + lamE*(logp_k - Weq)
            logp_k += yhat / 100.0; r2, r1 = r1, yhat
        err["arimax_ecm"][h].append(np.exp(logp_k) - lvl[o + h])
    if (o - WINDOW + 2) % 200 == 0: print(f"  {o - WINDOW + 2}/{n - WINDOW}")
print("ecm done")

def metrics(e, a):
    e = np.asarray(e, float); a = np.asarray(a, float)
    return np.mean(np.abs(e)), np.sqrt(np.mean(e**2)), 100 * np.mean(np.abs(e / a))

NAMES = [("rw","RW"),("rw_drift","RW-drift"),("arima","ARIMA-GARCH"),
         ("arimax","ARIMAX-GARCH"),("arimax_ecm","ARIMAX-ECM-GARCH")]
print("=" * 70); print("TABLE III  Point-forecast accuracy (MAE & RMSE in IDR, MAPE in %)"); print("=" * 70)
for h in H:
    a = np.asarray(act[h], float); print(f"\nHorizon h = {h}")
    print(f"  {'model':<18}{'MAE':>9}{'RMSE':>9}{'MAPE%':>9}")
    for key, lab in NAMES:
        mae, rmse, mape = metrics(err[key][h], a)
        print(f"  {lab:<18}{mae:>9.2f}{rmse:>9.2f}{mape:>9.4f}")

actuals = {h: np.array([lvl[o + h] for o in range(WINDOW - 1, n - 1) if o + h <= n - 1], float) for h in H}
err_ret = {m: {h: np.log1p(np.asarray(err[m][h], float) / actuals[h]) for h in H} for m in err}

def clark_west(e_small, e_large, h):
    e_small = np.asarray(e_small, float); e_large = np.asarray(e_large, float)
    f = e_small**2 - e_large**2 + (e_small - e_large)**2
    f = f[~np.isnan(f)]
    t = float(sm.OLS(f, np.ones((f.size, 1))).fit(cov_type="HAC",
              cov_kwds={"maxlags": max(h - 1, 0)}).tvalues[0])
    return float(1 - stats.norm.cdf(t))          # one-sided p-value

def cw_table(big, benches, title):
    print("\n" + title)
    print(f"{'Horizon':<9}" + "".join(f"{'vs ' + b:<14}" for b in benches))
    for h in H:
        cells = "".join(f"{clark_west(err_ret[b][h], err_ret[big][h], h):<14.3f}" for b in benches)
        print(f"{h:<9}{cells}")

cw_table("arimax", ["rw", "rw_drift", "arima"],
         "TABLE IV  Clark-West: ARIMAX-GARCH vs each benchmark (one-sided p-value)")
cw_table("arimax_ecm", ["rw", "rw_drift", "arima", "arimax"],
         "TABLE V  Clark-West: ARIMAX-ECM-GARCH vs each benchmark (one-sided p-value)")

S = 2000; Z = stats.norm.ppf(1 - ALPHA / 2); QLO, QHI = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
MODELS = ["rw", "rw_drift", "arima_garch", "arimax_garch", "arimax_frozen", "arimax_ecm_garch"]
band = {m: {h: {"lo": [], "hi": [], "act": []} for h in H} for m in MODELS}
def store(m, h, lo, hi, a): band[m][h]["lo"].append(lo); band[m][h]["hi"].append(hi); band[m][h]["act"].append(a)
def gstate(res, ne):
    p = res.params.to_numpy(); sig = np.asarray(res.conditional_volatility, float); sig = sig[~np.isnan(sig)]
    eps = np.asarray(res.resid, float); eps = eps[~np.isnan(eps)]
    return p[0], p[1], p[2], p[3+ne], p[4+ne], p[5+ne], eps[-1], sig[-1]**2
def sim(o, c, a1, a2, w, al, be, epsT, hT, h, garch=True, fvar=None):
    r1 = np.full(S, y[o]); r2 = np.full(S, y[o-1]); ep = np.full(S, epsT); hv = np.full(S, hT); s = np.zeros(S)
    for _ in range(h):
        hv = w + al*ep**2 + be*hv if garch else np.full(S, fvar)
        ep = np.sqrt(hv) * rng.standard_normal(S); rk = c + a1*r1 + a2*r2 + ep; s += rk; r2, r1 = r1, rk
    return lvl[o] * np.exp(s / 100.0)
def sim_ecm(o, c, a1, a2, lam, w, al, be, epsT, hT, a_ec, b_ec, h):
    r1 = np.full(S, y[o]); r2 = np.full(S, y[o-1]); ep = np.full(S, epsT); hv = np.full(S, hT)
    lp = np.full(S, logp[o]); Weq = a_ec + b_ec @ Wlvl[o]
    for _ in range(h):
        hv = w + al*ep**2 + be*hv; ep = np.sqrt(hv) * rng.standard_normal(S)
        yk = c + a1*r1 + a2*r2 + lam*(lp - Weq) + ep; lp = lp + yk/100.0; r2, r1 = r1, yk
    return np.exp(lp)

for o in range(WINDOW - 1, n - 1):
    tr = slice(o - WINDOW + 1, o + 1); sd = y[tr].std(ddof=1); mu = y[tr].mean()
    for h in H:
        if o + h > n - 1: continue
        store("rw", h, lvl[o]*np.exp(-Z*np.sqrt(h)*sd/100.0), lvl[o]*np.exp(Z*np.sqrt(h)*sd/100.0), lvl[o+h])
        store("rw_drift", h, lvl[o]*np.exp((h*mu-Z*np.sqrt(h)*sd)/100.0), lvl[o]*np.exp((h*mu+Z*np.sqrt(h)*sd)/100.0), lvl[o+h])
    m0 = ARX(y[tr], lags=[1,2], rescale=False); m0.volatility = GARCH(1,0,1); s0 = gstate(m0.fit(disp="off"), 0)
    mx = ARX(y[tr], lags=[1,2], x=Xz[tr], rescale=False); mx.volatility = GARCH(1,0,1); sx = gstate(mx.fit(disp="off"), 3)
    denom = 1 - sx[4] - sx[5]; vun = sx[3]/denom if denom > 1e-6 else np.var(y[tr])
    a_ec, b_ec, zw = fit_ec_term(logp[tr], Wlvl[tr]); zlag = np.empty_like(zw); zlag[0]=0.0; zlag[1:]=zw[:-1]
    ok = True
    try:
        me = ARX(y[tr], lags=[1,2], x=np.column_stack([Xz[tr], zlag]), rescale=False); me.volatility = GARCH(1,0,1)
        pe = me.fit(disp="off"); pp = pe.params.to_numpy(); cE,a1E,a2E,lamE = pp[0],pp[1],pp[2],pp[6]; wE,alE,beE = pp[7],pp[8],pp[9]
        sig = np.asarray(pe.conditional_volatility, float); sig = sig[~np.isnan(sig)]
        rs = np.asarray(pe.resid, float); rs = rs[~np.isnan(rs)]; epE, hTE = rs[-1], sig[-1]**2
    except Exception:
        ok = False
    for h in H:
        if o + h > n - 1: continue
        lo, hi = np.percentile(sim(o, *s0, h, garch=True), [QLO, QHI]); store("arima_garch", h, lo, hi, lvl[o+h])
        lo, hi = np.percentile(sim(o, *sx, h, garch=True), [QLO, QHI]); store("arimax_garch", h, lo, hi, lvl[o+h])
        lo, hi = np.percentile(sim(o, sx[0],sx[1],sx[2],sx[3],sx[4],sx[5],sx[6],sx[7], h, garch=False, fvar=vun), [QLO, QHI])
        store("arimax_frozen", h, lo, hi, lvl[o+h])
        if ok:
            lo, hi = np.percentile(sim_ecm(o, cE,a1E,a2E,lamE,wE,alE,beE,epE,hTE, a_ec, b_ec, h), [QLO, QHI])
        else:
            lo = lvl[o]*np.exp((h*mu-Z*np.sqrt(h)*sd)/100.0); hi = lvl[o]*np.exp((h*mu+Z*np.sqrt(h)*sd)/100.0)
        store("arimax_ecm_garch", h, lo, hi, lvl[o+h])
    if (o - WINDOW + 2) % 200 == 0: print(f"  {o - WINDOW + 2}/{n - WINDOW}")
print("interval simulation done")

def interval_score(lo, hi, a, alpha=ALPHA):
    lo, hi, a = map(lambda v: np.asarray(v, float), (lo, hi, a))
    return ((hi - lo) + (2/alpha)*(lo - a)*(a < lo) + (2/alpha)*(a - hi)*(a > hi)).mean()
def kupiec(viol, p=ALPHA):
    viol = np.asarray(viol, int); N = len(viol); x = viol.sum(); pi = x/N
    if x == 0 or x == N: return np.nan
    lr = -2*((N-x)*np.log(1-p)+x*np.log(p)) + 2*((N-x)*np.log(1-pi)+x*np.log(pi))
    return float(1 - stats.chi2.cdf(lr, 1))
def christoffersen(viol, p=ALPHA):
    v = np.asarray(viol, int); n00=n01=n10=n11=0
    for i in range(1, len(v)):
        a, b = v[i-1], v[i]
        n00 += (a==0 and b==0); n01 += (a==0 and b==1); n10 += (a==1 and b==0); n11 += (a==1 and b==1)
    x = v.sum(); N = len(v); pi = x/N
    if x == 0 or x == N: return np.nan
    lr_uc = -2*((N-x)*np.log(1-p)+x*np.log(p)) + 2*((N-x)*np.log(1-pi)+x*np.log(pi))
    pi01 = n01/(n00+n01) if (n00+n01)>0 else 0; pi11 = n11/(n10+n11) if (n10+n11)>0 else 0
    pio = (n01+n11)/(n00+n01+n10+n11)
    def L(pr,k1,k0): return 0.0 if pr<=0 or pr>=1 else k1*np.log(pr)+k0*np.log(1-pr)
    lr_ind = -2*(L(pio,n01+n11,n00+n10) - (L(pi01,n01,n00)+L(pi11,n11,n10)))
    return float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2))

LAB = {"rw":"RW","rw_drift":"RW-drift","arima_garch":"ARIMA-GARCH","arimax_garch":"ARIMAX-GARCH",
       "arimax_frozen":"ARIMAX-frozen","arimax_ecm_garch":"ARIMAX-ECM-GARCH"}
print("=" * 78); print("95% INTERVAL-FORECAST EVALUATION (nominal coverage = 95%)"); print("=" * 78)
for h in H:
    print(f"\nHorizon h = {h}")
    print(f"  {'model':<18}{'cov%':>7}{'width':>9}{'rel%':>8}{'IS':>9}{'Kupiec':>9}{'Christ':>9}")
    for m in MODELS:
        lo = np.array(band[m][h]["lo"]); hi = np.array(band[m][h]["hi"]); a = np.array(band[m][h]["act"])
        viol = ((a < lo) | (a > hi)).astype(int)
        cov = 100*(1 - viol.mean()); width = (hi - lo).mean(); rel = 100*((hi - lo)/a).mean()
        ks = kupiec(viol); cs = christoffersen(viol)
        print(f"  {LAB[m]:<18}{cov:>7.1f}{width:>9.1f}{rel:>8.2f}{interval_score(lo,hi,a):>9.1f}"
              f"{ks:>9.3f}{cs:>9.3f}")

# Figure 5: interval coverage and relative width by horizon
order = ["rw","rw_drift","arima_garch","arimax_garch","arimax_frozen","arimax_ecm_garch"]
mk = {"rw":"o","rw_drift":"s","arima_garch":"^","arimax_garch":"D","arimax_frozen":"v","arimax_ecm_garch":"x"}
cov, relw = {}, {}
for m in order:
    cov[m], relw[m] = [], []
    for h in H:
        lo = np.array(band[m][h]["lo"]); hi = np.array(band[m][h]["hi"]); a = np.array(band[m][h]["act"])
        v = ((a < lo) | (a > hi)).astype(int)
        cov[m].append(100*(1 - v.mean())); relw[m].append(100*((hi - lo)/a).mean())
x = np.arange(len(H)); fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
for m in order:
    a1.plot(x, cov[m], marker=mk[m], lw=1.3, ms=5, label=LAB[m])
    a2.plot(x, relw[m], marker=mk[m], lw=1.3, ms=5, label=LAB[m])
a1.axhline(95, ls="--", color="0.5", lw=1)
a1.set_title("Empirical coverage of the 95% interval"); a1.set_ylabel("Coverage (%)")
a2.set_title("Relative interval width"); a2.set_ylabel("Width / level (%)")
for ax in (a1, a2):
    ax.set_xticks(x); ax.set_xticklabels(H); ax.set_xlabel("Horizon (trading days)")
    ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
a1.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("figure5.png", dpi=200, bbox_inches="tight"); plt.show()

# Figure 1: level of the exchange rate
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(df.index, df["fx"], color="blue", lw=1.0)
ax.set_title("IDR/USD Exchange Rate (JISDOR), 2018-2026")
ax.set_xlabel("Date"); ax.set_ylabel("IDR per USD")
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.savefig("figure1.png", dpi=200); plt.show()

# Figure 2: daily log-returns and squared returns (volatility clustering)
dates = data.index
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
a1.plot(dates, y, color="red", lw=0.6); a1.axhline(0, color="0.5", lw=0.6)
a1.set_title("Daily log-returns of the IDR/USD rate (%)"); a1.set_ylabel("Return (%)")
a2.plot(dates, y**2, color="purple", lw=0.6)
a2.set_title("Squared daily returns (volatility clustering)"); a2.set_ylabel("Squared return (%^2)"); a2.set_xlabel("Date")
a2.xaxis.set_major_locator(mdates.YearLocator()); a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.savefig("figure2.png", dpi=200); plt.show()

# Figure 3: last 250 one-day-ahead forecasts, actual vs ARIMAX-GARCH vs RW
f3_idx = list(range(WINDOW, n))
f3_actual = np.array(act[1]); f3_arimax = np.array(act[1]) - np.array(err["arimax"][1]); f3_rw = np.array(act[1]) - np.array(err["rw"][1])
k = min(250, len(f3_idx)); sel = slice(len(f3_idx) - k, len(f3_idx)); d3 = data.index[np.array(f3_idx)[sel]]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(d3, np.array(f3_actual)[sel], color="blue", lw=1.4, label="Actual")
ax.plot(d3, np.array(f3_arimax)[sel], color="green", lw=1.0, label="ARIMAX-GARCH")
ax.plot(d3, np.array(f3_rw)[sel], color="orange", lw=1.0, ls="--", label="Random walk")
ax.set_title("One-day-horizon rolling forecasts: actual vs ARIMAX-GARCH vs random walk (last 250 days)")
ax.set_xlabel("Date"); ax.set_ylabel("IDR/USD"); ax.legend(frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); fig.autofmt_xdate()
fig.savefig("figure3.png", dpi=200); plt.show()

# Figure 4: MAE reduction of ARIMAX-GARCH vs three benchmarks by horizon
red_rw, red_rwd, red_ar = [], [], []
for h in H:
    a = np.asarray(act[h], float)
    mae_x = np.mean(np.abs(err["arimax"][h])); mae_r = np.mean(np.abs(err["rw"][h]))
    mae_rwd = np.mean(np.abs(err["rw_drift"][h])); mae_a = np.mean(np.abs(err["arima"][h]))
    red_rw.append(100*(mae_r-mae_x)/mae_r); red_rwd.append(100*(mae_rwd-mae_x)/mae_rwd); red_ar.append(100*(mae_a-mae_x)/mae_a)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.axhline(0, color="0.5", lw=0.8)
ax.plot(H, red_rw, "-o", color="#1f4e79", label="ARIMAX-GARCH vs driftless random walk")
ax.plot(H, red_rwd, "-s", color="#e07b00", label="ARIMAX-GARCH vs random walk with drift")
ax.plot(H, red_ar, "-^", color="#4c9a2a", label="ARIMAX-GARCH vs ARIMA-GARCH (macro contribution)")
for xv, vv in zip(H, red_rw):
    ax.annotate(f"{vv:.1f}%", (xv, vv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#1f4e79")
ax.set_title("MAE reduction of ARIMAX-GARCH vs forecast horizon")
ax.set_xlabel("Forecast horizon (trading days)"); ax.set_ylabel("MAE reduction (%)")
ax.set_xticks(H); ax.legend(frameon=False, loc="best"); ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
fig.savefig("figure4.png", dpi=200, bbox_inches="tight"); plt.show()
