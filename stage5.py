"""Stage 5 robustness for islamic_malaysia_transmission (run after replicate.py).

Checks on the primary tests and the deposit-rate spreads:
  no_covid      drop windows touching 2020-03..2021-12 (loan repayment moratorium distorts rates on outstanding)
  no_2022_23    drop windows touching 2022-05..2023-12 (the post-pandemic hiking cycle)
  lags6         six own lags instead of three
  value_weight  credit panel weighted by each purpose's share of system financing (credit only)
Driscoll-Kraay / Newey-West errors; permutation p for the primary tests. Writes outputs/stage5_robustness.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import replicate as R

HS = [6, 12, 24]
WINDOWS = {"no_covid": ("2020-03", "2021-12"), "no_2022_23": ("2022-05", "2023-12")}
rows = []


def mask(s, w):
    a, b = pd.Period(w[0], "M"), pd.Period(w[1], "M")
    return s.where((s.index < a) | (s.index > b))


def add(check, block, unit, shock, h, res, perm=False):
    if res is None:
        return
    r = dict(check=check, block=block, unit=unit, shock=shock, h=h, b=res["b"], se=res["se"], n=res["n"])
    if perm:
        pc = R.CFG["inference"]["permutation"]
        r["p_perm"] = R.permute(res, shock, pc["draws"], pc["block_months"], pc["seed"] + h)
    rows.append(r)


# ---------------- pricing: book (primary) and deposit spreads
Rt = R.rates_series()
pairs = R.CFG["blocks"]["pricing"]["pairs"]
for name in ["book", "fd1", "fd3", "fd12", "savings"]:
    ri, rc = pairs[name]
    sp = (Rt[ri] - Rt[rc]).loc["2011-01":]
    for h in HS:
        for s in ["d_opr", "jk_mp"]:
            prim = name == "book"
            for chk, w in WINDOWS.items():
                add(chk, "pricing", name, s, h, R.ts_fit(mask(sp, w), s, h, step_dummies=("2015-01", "2015-07")), prim)
            R.LAGS = 6
            add("lags6", "pricing", name, s, h, R.ts_fit(sp, s, h, step_dummies=("2015-01", "2015-07")), prim)
            R.LAGS = 3
print("pricing done", flush=True)

# ---------------- pass-through levels without the moratorium
for name in ["book", "fd12"]:
    for side, col in zip(["islamic", "conventional"], pairs[name]):
        for h in HS:
            add("no_covid", "passthrough", f"{name}_{side}", "d_opr", h, R.ts_fit(mask(Rt[col].loc["2011-01":], WINDOWS["no_covid"]), "d_opr", h))

# ---------------- credit panel (primary)
cats = R.HEADLINE_PURPOSES
ser = R.financing_series("purpose", cats)
fin = pd.read_csv(R.CLEAN / "financing.csv")
fin = fin[(fin.dim == "purpose") & (fin.bank == "system") & fin.category.isin(cats)]
fin["month"] = pd.PeriodIndex(fin["month"], freq="M")
share = fin.pivot_table(index="month", columns="category", values="value")
share = share.div(share.sum(axis=1), axis=0).shift(1)          # lagged system share of each purpose

for h in HS:
    for s in ["jk_mp", "d_opr"]:
        for chk, w in WINDOWS.items():
            sw = {k: (mask(y, w), bad) for k, (y, bad) in ser.items()}
            add(chk, "credit", "purpose", s, h, R.fit_panel(R.panel_frames(sw, cats, h), s, h=h), True)
        add("lags6", "credit", "purpose", s, h, R.fit_panel(R.panel_frames(ser, cats, h, lags=6), s, lags=6, h=h), True)
        # value-weighted WLS with Driscoll-Kraay errors
        d = R.panel_frames(ser, cats, h).merge(R.SHOCKS[[s]], left_on="month", right_index=True)
        d["w"] = [share.at[m, c] if m in share.index else np.nan for m, c in zip(d["month"], d["cat"])]
        cols = [s] + [f"lag{l}" for l in range(1, 4)]
        d = d.dropna(subset=["dep", "w"] + cols)
        X = pd.concat([d[cols], pd.get_dummies(d["cat"], drop_first=True, dtype=float)], axis=1)
        X.insert(0, "const", 1.0)
        t = d["month"].astype("int64").values
        o = np.argsort(t, kind="stable")
        m = sm.WLS(d["dep"].values[o], X.values[o], weights=d["w"].values[o]).fit(
            cov_type="hac-groupsum", cov_kwds={"time": t[o], "maxlags": h + 2})
        rows.append(dict(check="value_weight", block="credit", unit="purpose", shock=s, h=h, b=m.params[1],
                         se=m.bse[1], n=len(d)))
print("credit done", flush=True)

pd.DataFrame(rows).to_csv(R.OUT / "stage5_robustness.csv", index=False)
print("wrote outputs/stage5_robustness.csv")
