"""islamic_malaysia_transmission: Islamic minus conventional local projections, Malaysia.

Credit (H3) and SME (H5) panels: with purpose x month and purpose x bank fixed effects and two bank types per
purpose-month, the regression is identical to one on the Islamic-minus-conventional difference within purpose:

    D_h(p,t) = [y_I(p,t+h) - y_I(p,t-1)] - [y_C(p,t+h) - y_C(p,t-1)]
             = b_h e_t + sum_l r_l [dy_I(p,t-l) - dy_C(p,t-l)] + a_p + u

so b_h is the extra Islamic response. Driscoll-Kraay errors (groups = month). Pricing (H1, H2a, H4) and funding mix
(H2b): single time series of the Islamic-minus-conventional spread / share, Newey-West errors. All blocks: 12-month
block permutation of the shock (Frisch-Waugh) and a pre-shock placebo.

Run from this folder with MaRCo's Python:  <MaRCo>/.venv/Scripts/python.exe replicate.py
Reads config.yaml and data/clean/; writes outputs/.
"""
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

HERE = pathlib.Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
CLEAN = HERE / "data" / "clean"
OUT = HERE / "outputs"
SPLICE = pd.Period("2021-07", "M")
HEADLINE_PURPOSES = ["residential", "non_residential", "transport", "personal", "credit_card", "construction",
                     "working_capital", "fixed_assets_other"]
HEADLINE_SECTORS = ["agriculture", "mining", "manufacturing", "utilities", "trade_hotels", "construction",
                    "real_estate", "transport_ict", "finance_business", "education_health", "household"]
LAGS = 3
H = CFG["horizons"]["max"]


# ---------------------------------------------------------------- shocks
def load_shocks():
    s = pd.read_csv(CLEAN / "shocks.csv")
    s["month"] = pd.PeriodIndex(s["month"], freq="M")
    s = s.set_index("month")
    span = s.loc["2006-12":"2026-09"]
    out = pd.DataFrame(index=s.index)
    out["jk_mp"] = s["jk_mp"] / span["jk_mp"].std()          # per 1 s.d.
    out["d_opr"] = s["d_opr"] / 0.25                         # per +25 bp
    out["vix_innov"] = s["vix_innov"] / span["vix_innov"].std()
    out["d_opr_raw"] = s["d_opr"]
    sd = {"jk_mp_sd_pp": float(span["jk_mp"].std()), "vix_innov_sd": float(span["vix_innov"].std())}
    return out, sd


SHOCKS, SHOCK_SD = load_shocks()
SHOCK_NAMES = list(CFG["shocks"])


# ---------------------------------------------------------------- series preparation
def prepare(series, bad=None):
    """series: pd.Series (monthly PeriodIndex) of an outcome in levels (already transformed).
    bad: boolean Series, True where the change INTO that month is unusable. Returns (y, dy, cumulative bad)."""
    idx = pd.period_range(series.index.min(), series.index.max(), freq="M")
    y = series.reindex(idx)
    dy = y.diff()
    b = dy.isna()
    if bad is not None:
        b |= bad.reindex(idx).fillna(False).astype(bool)
    dy = dy.where(~b)
    return y, dy, b.astype(int).cumsum()


def lp_frame(y, dy, cb, h, lags=LAGS):
    """Dependent variable y(t+h) - y(t-1) on clean windows, lags of dy, placebo y(t-1) - y(t-5)."""
    f = pd.DataFrame(index=y.index)
    f["dep"] = y.shift(-h) - y.shift(1)
    f.loc[(cb.shift(-h) - cb.shift(1)) != 0, "dep"] = np.nan
    for l in range(1, lags + 1):
        f[f"lag{l}"] = dy.shift(l)
    f["pre"] = y.shift(1) - y.shift(5)
    f.loc[(cb.shift(1) - cb.shift(5)) != 0, "pre"] = np.nan
    return f


# ---------------------------------------------------------------- credit and SME panels
def financing_series(dim, cats):
    f = pd.read_csv(CLEAN / "financing.csv")
    f = f[(f.dim == dim) & f.category.isin(cats) & f.bank.isin(["islamic", "conventional"])]
    f["month"] = pd.PeriodIndex(f["month"], freq="M")
    ratio = pd.read_csv(CLEAN / "splice_ratio.csv")
    ratio = ratio[ratio.dim == dim].set_index(["category", "bank"]).ratio
    br = CFG["breaks"]
    tol = 0.05
    out = {}
    for (c, b), g in f.groupby(["category", "bank"]):
        s = g.set_index("month").value.sort_index()
        s = s[s > 0]
        bad = pd.Series(False, index=s.index)
        r = ratio.get((c, b), np.nan)
        if np.isfinite(r) and abs(r - 1) <= tol:
            s = s.where(s.index >= SPLICE, s * r)       # chain-link the old series onto the revised one
        else:
            bad.loc[bad.index == SPLICE] = True         # windows crossing the splice are dropped
        y = 100 * np.log(s)
        if b == "islamic":                               # 2018-04: an institution entering the Islamic system
            for e in br["events"]:
                bad.loc[bad.index == pd.Period(e["month"], "M")] = True
        jump = y.diff().abs() > 10                       # jump rule
        out[(c, b)] = (y, bad | jump.reindex(bad.index).fillna(False))
    return out


def panel_frames(series, cats, h, lags=LAGS):
    rows = []
    for c in cats:
        if (c, "islamic") not in series or (c, "conventional") not in series:
            continue
        fr = {}
        for b in ["islamic", "conventional"]:
            y, bad = series[(c, b)]
            yy, dy, cb = prepare(y, bad)
            fr[b] = lp_frame(yy, dy, cb, h, lags)
        d = fr["islamic"] - fr["conventional"]
        d["cat"] = c
        rows.append(d.dropna(subset=["dep"]))
    return pd.concat(rows).rename_axis("month").reset_index()


def fit_panel(d, shock, dep="dep", lags=LAGS, h=0, with_lags=True):
    d = d.merge(SHOCKS[[shock]], left_on="month", right_index=True, how="left")
    cols = [shock] + ([f"lag{l}" for l in range(1, lags + 1)] if with_lags else [])
    d = d.dropna(subset=[dep] + cols)
    if len(d) < 60:
        return None
    fe = pd.get_dummies(d["cat"], prefix="c", drop_first=True, dtype=float)
    X = pd.concat([d[cols], fe], axis=1)
    X.insert(0, "const", 1.0)
    t = d["month"].astype("int64").values
    o = np.argsort(t, kind="stable")
    m = sm.OLS(d[dep].values[o], X.values[o]).fit(cov_type="hac-groupsum", cov_kwds={"time": t[o], "maxlags": h + 2})
    return dict(b=m.params[1], se=m.bse[1], n=len(d), months=d["month"].nunique(), units=d["cat"].nunique(),
                _X=X.values[o], _y=d[dep].values[o], _t=d["month"].values[o])


# ---------------------------------------------------------------- time-series blocks
def rates_series():
    r = pd.read_csv(CLEAN / "rates.csv")
    r["month"] = pd.PeriodIndex(r["month"], freq="M")
    return r.set_index("month")


def deposit_shares():
    d = pd.read_csv(CLEAN / "deposits.csv")
    d["month"] = pd.PeriodIndex(d["month"], freq="M")
    t = d[d.dim == "type"].pivot_table(index="month", columns=["bank", "category"], values="value")
    isl = t["islamic"].drop(columns=["term"]).rename(columns={"term_incl_ia": "term"})
    conv = t["conventional"].drop(columns=[c for c in ["term_incl_ia"] if c in t["conventional"]])
    out = {}
    for c in ["demand", "savings", "term"]:
        out[c] = 100 * isl[c] / isl.sum(axis=1) - 100 * conv[c] / conv.sum(axis=1)
    return pd.DataFrame(out)


def ts_fit(series, shock, h, extra_controls=True, dep="dep", with_lags=True, step_dummies=(), pre_controls=False):
    y, dy, cb = prepare(series.dropna())
    f = lp_frame(y, dy, cb, h, LAGS)
    f = f.join(SHOCKS[[shock]], how="left")
    f["dopr"] = SHOCKS["d_opr"].reindex(f.index)
    cols = [shock] + ([f"lag{l}" for l in range(1, LAGS + 1)] if with_lags else [])
    if extra_controls and with_lags:
        for l in range(1, LAGS + 1):
            f[f"dopr{l}"] = f["dopr"].shift(l)
            cols.append(f"dopr{l}")
    if pre_controls:   # conditional placebo: the policy-rate changes inside the pre-window (t-4..t-1)
        for l in range(1, 5):
            f[f"dopr{l}"] = f["dopr"].shift(l)
            cols.append(f"dopr{l}")
    for m in step_dummies:                       # level shift in month m falls inside (t-1, t+h]
        p = pd.Period(m, "M")
        k = f"step_{m}"
        f[k] = ((f.index <= p) & (f.index + h >= p)).astype(float)
        cols.append(k)
    f = f.dropna(subset=[dep] + cols)
    cols = [c for c in cols if not (c.startswith("step_") and f[c].std() == 0)]   # no variation in this sample
    if len(f) < 40:
        return None
    X = sm.add_constant(f[cols].values)
    m = sm.OLS(f[dep].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 2})
    return dict(b=m.params[1], se=m.bse[1], n=len(f), months=len(f), units=1, _X=X, _y=f[dep].values,
                _t=f.index.values)


# ---------------------------------------------------------------- permutation
def permute(res, shock, n, block, seed):
    """Block-permute the shock across calendar months; b re-estimated by Frisch-Waugh (column 1 of _X)."""
    X, y, t = res["_X"], res["_y"], res["_t"]
    W = np.delete(X, 1, axis=1)
    Q, _ = np.linalg.qr(W)
    resid = lambda a: a - Q @ (Q.T @ a)
    yr = resid(y)
    base = SHOCKS[shock].loc["2006-12":"2026-09"].dropna()
    pos = {p: i for i, p in enumerate(base.index)}
    keep = np.array([p in pos for p in t])
    T = len(base)
    blocks = [np.arange(k, min(k + block, T)) for k in range(0, T, block)]
    rowpos = np.array([pos.get(p, 0) for p in t])
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        idx = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])[:T]
        e = base.values[idx][rowpos]
        e = np.where(keep, e, 0.0)
        x = resid(e)
        draws[i] = (x @ yr) / (x @ x)
    return float(np.mean(np.abs(draws) >= abs(res["b"])))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- run
def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    OUT.mkdir(exist_ok=True)
    pc = CFG["inference"]["permutation"]
    perm_h = [6, 12, 24]
    rows, pre_rows = [], []

    def record(block, unit, shock, h, res, permute_it):
        if res is None:
            return
        r = dict(block=block, unit=unit, shock=shock, h=h, **strip(res))
        if permute_it:
            r["p_perm"] = permute(res, shock, pc["draws"], pc["block_months"], pc["seed"] + h)
        rows.append(r)

    # credit panels (H3)
    for dim, cats in [("purpose", HEADLINE_PURPOSES), ("sector", HEADLINE_SECTORS)]:
        ser = financing_series(dim, cats)
        for h in range(H + 1):
            d = panel_frames(ser, cats, h)
            for s in SHOCK_NAMES:
                record("credit", dim, s, h, fit_panel(d, s, h=h), h in perm_h)
                if h == 0:
                    p = fit_panel(d.rename(columns={"dep": "x", "pre": "dep"}), s, h=4, with_lags=False)
                    if p:
                        pre_rows.append(dict(block="credit", unit=dim, shock=s, **strip(p)))
        print(f"credit {dim} done", flush=True)

    # pricing (H1, H2a, H4) and pass-through levels
    R = rates_series()
    pairs = CFG["blocks"]["pricing"]["pairs"]
    for name, (ri, rc) in pairs.items():
        spread = (R[ri] - R[rc]).loc["2011-01":]
        for h in range(H + 1):
            for s in SHOCK_NAMES:
                record("pricing", name, s, h, ts_fit(spread, s, h, step_dummies=("2015-01", "2015-07")), h in perm_h)
                if h == 0:
                    for kind, pcf in [("unconditional", False), ("conditional", True)]:
                        p = ts_fit(spread, s, 4, dep="pre", with_lags=False, pre_controls=pcf)
                        if p:
                            pre_rows.append(dict(block="pricing", unit=name, shock=s, placebo=kind, **strip(p)))
        print(f"pricing {name} done", flush=True)
    for name, (ri, rc) in pairs.items():
        for side, col in [("islamic", ri), ("conventional", rc)]:
            lev = R[col].loc["2011-01":]
            for h in [0, 3, 6, 12, 18, 24]:
                record("passthrough", f"{name}_{side}", "d_opr", h, ts_fit(lev, "d_opr", h), False)

    # pass-through as a share of the OPR's own cumulative path (OPR moves come in sequences, so one move
    # predicts further moves; the ratio is the economically meaningful pass-through)
    sh = pd.read_csv(CLEAN / "shocks.csv")
    sh["month"] = pd.PeriodIndex(sh["month"], freq="M")
    opr = sh.set_index("month")["opr"].loc["2011-01":"2026-09"]
    ratio_rows = []
    for h in [0, 3, 6, 12, 18, 24]:
        o = ts_fit(opr, "d_opr", h, extra_controls=False)
        row = {"h": h, "opr_path_pp_per_25bp": o["b"]}
        for name, (ri, rc) in pairs.items():
            for side, col in [("islamic", ri), ("conventional", rc)]:
                r = ts_fit(R[col].loc["2011-01":], "d_opr", h)
                if r:
                    row[f"{name}_{side}"] = r["b"] / o["b"]
        ratio_rows.append(row)
    pd.DataFrame(ratio_rows).to_csv(OUT / "passthrough_ratio.csv", index=False)

    # funding mix (H2b): Islamic minus conventional share of funding, pp
    S = deposit_shares()
    for c in S:
        for h in range(H + 1):
            for s in SHOCK_NAMES:
                record("funding", c, s, h, ts_fit(S[c], s, h, step_dummies=("2015-07",)), h in perm_h)
                if h == 0:
                    for kind, pcf in [("unconditional", False), ("conditional", True)]:
                        p = ts_fit(S[c], s, 4, dep="pre", with_lags=False, pre_controls=pcf)
                        if p:
                            pre_rows.append(dict(block="funding", unit=c, shock=s, placebo=kind, **strip(p)))
    print("funding done", flush=True)

    # SMEs (H5, exploratory): approval rate (pp) and log disbursed / outstanding, sector x bank, 2021-07 on
    sme = pd.read_csv(CLEAN / "sme.csv")
    sme["month"] = pd.PeriodIndex(sme["month"], freq="M")
    sme = sme[sme.sector != "total"]
    sme["approval_rate"] = 100 * sme["approved"] / sme["applied"].where(sme["applied"] > 0)
    sme.loc[sme["applied"] < 20, "approval_rate"] = np.nan       # tiny application volumes give noisy ratios
    for var in ["approval_rate", "outstanding"]:
        ser = {}
        for (c, b), g in sme.groupby(["sector", "bank"]):
            s = g.set_index("month")[var].sort_index()
            if var == "outstanding":
                s = 100 * np.log(s[s > 0])
            ser[(c, b)] = (s.dropna(), None)
        cats = sorted(sme.sector.unique())
        for h in range(0, 13):
            d = panel_frames(ser, cats, h, lags=1)
            for s in SHOCK_NAMES:
                record("sme", var, s, h, fit_panel(d, s, lags=1, h=h), False)
    print("sme done", flush=True)

    res = pd.DataFrame(rows)
    pre = pd.DataFrame(pre_rows)
    # primary tests, Holm on permutation p-values
    prim = []
    for t in CFG["inference"]["primary_tests"]:
        unit = t.get("dim", t.get("pair"))
        r = res[(res.block == t["block"]) & (res.unit == unit) & (res.shock == t["shock"]) & (res.h == t["h"])]
        prim.append(dict(block=t["block"], unit=unit, shock=t["shock"], h=t["h"], b=r.b.iloc[0], se=r.se.iloc[0],
                         p_perm=r.p_perm.iloc[0]))
    prim = pd.DataFrame(prim)
    prim["p_holm"] = holm(prim.p_perm.values)
    res.to_csv(OUT / "lp_results.csv", index=False)
    pre.to_csv(OUT / "lp_placebo.csv", index=False)
    prim.to_csv(OUT / "primary_tests.csv", index=False)
    s = SHOCKS.loc["2006-12":"2026-09"]
    diag = {"shock_sd": SHOCK_SD,
            "shock_ar1": {k: float(s[k].dropna().autocorr()) for k in SHOCK_NAMES},
            "corr": s[SHOCK_NAMES].corr().round(3).to_dict(),
            "opr_moves": int((s["d_opr_raw"].abs() > 0).sum())}
    (OUT / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((HERE / "config.yaml").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    print("wrote outputs/")


if __name__ == "__main__":
    main()
