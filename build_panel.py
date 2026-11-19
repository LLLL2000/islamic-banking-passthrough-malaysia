"""Build the clean monthly panels for islamic_malaysia_transmission from data/raw/.

Run from this folder:  <MaRCo>/.venv/Scripts/python.exe -I build_panel.py   (needs bnm_parse.py alongside)
Writes data/clean/{financing,deposits,rates,sme,shocks}.csv and data/clean/checks/*.csv.

Conventional = whole banking system minus the Islamic banking system (Islamic banks + Islamic windows of
commercial and investment banks). Column positions below are read off each table's header (see bnm_parse.py);
they are fixed within a file and are verified by the identity and splice checks at the end.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bnm_parse as B  # noqa: E402

RAW = HERE / "data" / "raw"
CLEAN = HERE / "data" / "clean"
CHK = CLEAN / "checks"
SPLICE = pd.Period("2021-07", "M")   # first month of the revised (MSIC 2008-based) financing tables
SPLICE_RATIOS = []

# ---------------------------------------------------------------- harmonised categories: {name: [columns]}
PURPOSE = {  # old (pre-2021-07) tables
    "old_islamic": {"securities": [3], "transport": [4], "residential": [6], "non_residential": [7],
                    "fixed_assets_other": [8], "personal": [9, 11], "credit_card": [10], "construction": [12],
                    "working_capital": [13], "other": [14], "total": [15]},
    "old_total":   {"securities": [3], "transport": [4], "residential": [6], "non_residential": [7],
                    "fixed_assets_other": [8], "personal": [9, 11], "credit_card": [10], "construction": [12],
                    "working_capital": [13], "other": [14], "total": [15]},
    # new tables: transport vehicles sit under 'fixed assets other than land and building'; personal uses
    # total = consumer durables + others (old 'personal use' + 'consumer durables')
    "new_islamic": {"securities": [3], "transport": [5], "residential": [9], "non_residential": [10],
                    "fixed_assets_other": [8], "personal": [11], "credit_card": [14], "construction": [15],
                    "working_capital": [16], "other": [17], "total": [18]},
    "new_total":   {"securities": [3], "transport": [5], "residential": [9], "non_residential": [15],
                    "fixed_assets_other": [8], "personal": [21], "credit_card": [24], "construction": [25],
                    "working_capital": [26], "other": [27], "total": [28]},
}
SECTOR = {
    "old_islamic": {"agriculture": [16], "mining": [17], "manufacturing": [18], "utilities": [19],
                    "trade_hotels": [20], "construction": [21], "real_estate": [22], "transport_ict": [23],
                    "finance_business": [24], "education_health": [25], "household": [26], "other": [27]},
    "old_total":   {"agriculture": [3], "mining": [4], "manufacturing": [5], "utilities": [6],
                    "trade_hotels": [7], "construction": [8], "real_estate": [9], "transport_ict": [10],
                    "finance_business": [11], "education_health": [12], "household": [13], "other": [14],
                    "total": [15]},
    "new_islamic": {"agriculture": [19], "mining": [20], "manufacturing": [21], "utilities": [22, 23],
                    "trade_hotels": [25, 29], "construction": [24], "real_estate": [34], "transport_ict": [30, 31],
                    "finance_business": [33, 35, 36], "education_health": [37], "household": [38], "other": [39],
                    "total": [40]},
    "new_total":   {"agriculture": [3], "mining": [12], "manufacturing": [18], "utilities": [40, 41],
                    "trade_hotels": [49, 53], "construction": [42], "real_estate": [58], "transport_ict": [54, 55],
                    "finance_business": [57, 59, 60], "education_health": [61], "household": [62], "other": [63],
                    "total": [64]},
}
SECTOR["old_islamic"]["total"] = [15]
# deposits by type: Islamic 1.24.2 (per-type 'Total' columns), whole system 1.24
DEP_TYPE = {
    "islamic": {"demand": [42], "savings": [54], "term": [10, 26, 74], "nid": [66],
                "fx": [18, 34, 50, 58, 82, 98], "other": [90]},
    "total":   {"demand": [2], "savings": [4], "term": [3, 7], "nid": [5], "fx": [6], "other": [8]},
}
HOLDERS = {"government": ["Federal Government", "State Government", "Statutory Agency"],
           "financial_institution": ["Financial Institution"], "business": ["Business Enterprises"],
           "individuals": ["Individuals"], "other": ["Others"]}
DEP_HOLDER_TOTAL = {"government": [2, 3, 4], "financial_institution": [5], "business": [6], "individuals": [7],
                    "other": [8]}
# rates: 2.1 commercial banks (conventional), 2.2 Islamic banks and IBS of commercial banks
RATES = {  # simple averages: the weighted-average columns of 2.1/2.2 start only in 2026
    "conv_alr": ("2.1", 19), "conv_blr": ("2.1", 17), "conv_br": ("2.1", 16),
    "conv_fd1": ("2.1", 3), "conv_fd3": ("2.1", 4), "conv_fd6": ("2.1", 5), "conv_fd12": ("2.1", 7),
    "conv_sav": ("2.1", 13),
    "isl_afr": ("2.2", 24), "isl_bfr": ("2.2", 22), "isl_br": ("2.2", 20),
    "isl_fd1": ("2.2", 3), "isl_fd3": ("2.2", 4), "isl_fd6": ("2.2", 5), "isl_fd12": ("2.2", 7),
    "isl_sav": ("2.2", 18), "isl_ia12": ("2.2", 17),
}
SME_SECTOR = {"agriculture": [3], "mining": [4], "manufacturing": [5], "utilities": [6, 7], "construction": [8],
              "trade": [9], "accommodation_food": [13], "transport": [14], "ict": [15], "finance": [16],
              "real_estate": [17], "professional": [18], "admin_support": [19], "education_health": [20],
              "other": [21], "total": [22]}


def tbl(t):
    return B.read_table(RAW / "bnm" / f"{t}.xlsx")[0]


def collapse(L, mapping, inst):
    """Sum mapped columns for one institution row label -> wide DataFrame month x category."""
    x = L[L.inst == inst]
    out = {}
    for name, cols in mapping.items():
        s = x[x.col.isin(cols)].groupby("month").value.agg(lambda v: v.sum(min_count=len(cols)) if len(v) == len(cols) else np.nan)
        out[name] = s
    return pd.DataFrame(out)


def financing():
    rows, checks = [], []
    specs = [("purpose", "1.19.1a", "1.19.1", "1.19a", "1.19", PURPOSE),
             ("sector", "1.19.1a", "1.19.1", "1.20a", "1.20", SECTOR)]
    for dim, isl_old, isl_new, tot_old, tot_new, M in specs:
        io = collapse(tbl(isl_old), M["old_islamic"], "Total")
        inew = collapse(tbl(isl_new), M["new_islamic"], "Total")
        to = collapse(tbl(tot_old), M["old_total"], "Total")
        tnew = collapse(tbl(tot_new), M["new_total"], "Total")
        # splice check: ratio new/old in overlap months, per category
        for name in M["new_islamic"]:
            for side, o, n in [("islamic", io, inew), ("total", to, tnew)]:
                both = pd.concat([o[name], n[name]], axis=1, keys=["old", "new"]).dropna()
                if len(both):
                    r = both.new / both.old
                    checks.append(dict(dim=dim, category=name, side=side, overlap_months=len(both),
                                       ratio_mean=round(r.mean(), 4), ratio_min=round(r.min(), 4), ratio_max=round(r.max(), 4)))
        isl = pd.concat([io[io.index < SPLICE], inew[inew.index >= SPLICE]])
        tot = pd.concat([to[to.index < SPLICE], tnew[tnew.index >= SPLICE]])
        conv = tot - isl.reindex(tot.index)
        # new/old ratio in the first revised month, per bank side (conventional from old system minus old Islamic)
        if SPLICE in io.index and SPLICE in to.index:
            old = {"islamic": io.loc[SPLICE], "system": to.loc[SPLICE], "conventional": to.loc[SPLICE] - io.loc[SPLICE]}
            new = {"islamic": inew.loc[SPLICE], "system": tnew.loc[SPLICE], "conventional": tnew.loc[SPLICE] - inew.loc[SPLICE]}
            for bank in old:
                for name in M["new_islamic"]:
                    SPLICE_RATIOS.append(dict(dim=dim, category=name, bank=bank,
                                              ratio=float(new[bank][name] / old[bank][name])))
        for bank, w in [("islamic", isl), ("conventional", conv), ("system", tot)]:
            long = w.rename_axis("month").reset_index().melt("month", var_name="category", value_name="value")
            long["dim"], long["bank"] = dim, bank
            rows.append(long)
    f = pd.concat(rows).dropna(subset=["value"])
    return f[["month", "dim", "category", "bank", "value"]], pd.DataFrame(checks)


def identity_checks():
    """Do the whole-system tables add up, and does their 'Islamic banks' row match the Islamic table's?"""
    out = []
    for t_tot, t_isl, col_tot, col_isl in [("1.19a", "1.19.1a", 15, 15), ("1.19", "1.19.1", 28, 18)]:
        T = tbl(t_tot)
        I = tbl(t_isl)
        tt = T[T.col == col_tot].pivot_table(index="month", columns="inst", values="value")
        ii = I[I.col == col_isl].pivot_table(index="month", columns="inst", values="value")
        parts = [c for c in tt.columns if c not in ("Total", "Commercial banks")] if "Commercial banks and Islamic banks" in tt else [c for c in tt.columns if c != "Total"]
        if "Commercial banks and Islamic banks" in tt:
            parts = ["Commercial banks and Islamic banks", "Merchant or Investment banks"]
        gap = (tt[parts].sum(axis=1) - tt["Total"]) / tt["Total"]
        ib_tot = tt.get("Islamic banks")
        ib_isl = ii.get("Islamic banks", ii.get("Islamic Banks"))
        if ib_tot is None:
            ib_tot = tt.get("Islamic Banks")
        m = pd.concat([ib_tot, ib_isl], axis=1, keys=["in_system_table", "in_islamic_table"]).dropna()
        out.append(dict(tables=f"{t_tot} vs {t_isl}", months=len(tt),
                        sum_of_parts_vs_total_max_abs_gap=float(gap.abs().max()),
                        islamic_banks_row_ratio_mean=float((m.in_system_table / m.in_islamic_table).mean()) if len(m) else np.nan,
                        islamic_banks_row_ratio_min=float((m.in_system_table / m.in_islamic_table).min()) if len(m) else np.nan,
                        islamic_banks_row_ratio_max=float((m.in_system_table / m.in_islamic_table).max()) if len(m) else np.nan,
                        islamic_system_share_first=float((ii["Total"] / tt["Total"]).dropna().iloc[0]),
                        islamic_system_share_last=float((ii["Total"] / tt["Total"]).dropna().iloc[-1])))
    return pd.DataFrame(out)


def deposits():
    I = tbl("1.24.2")
    T = tbl("1.24")
    H = tbl("1.25")
    isl = collapse(I, DEP_TYPE["islamic"], "Jumlah/ Total")
    tot = collapse(T, DEP_TYPE["total"], "")
    rows = []
    for bank, w in [("islamic", isl), ("conventional", tot - isl.reindex(tot.index)), ("system", tot)]:
        long = w.rename_axis("month").reset_index().melt("month", var_name="category", value_name="value")
        long["dim"], long["bank"] = "type", bank
        rows.append(long)
    # holders: Islamic = sum over deposit types of each holder column; system from 1.25 (includes repos)
    lab = B.read_table(RAW / "bnm" / "1.24.2.xlsx")[1]
    x = I[I.inst == "Jumlah/ Total"].copy()
    x["holder"] = x.col.map(lambda j: lab[j].split(" | ")[-1])
    ih = {g: x[x.holder.isin(v)].groupby("month").value.sum() for g, v in HOLDERS.items()}
    ih = pd.DataFrame(ih)
    th = collapse(H, DEP_HOLDER_TOTAL, "")
    for bank, w in [("islamic", ih), ("conventional", th - ih.reindex(th.index)), ("system", th)]:
        long = w.rename_axis("month").reset_index().melt("month", var_name="category", value_name="value")
        long["dim"], long["bank"] = "holder", bank
        rows.append(long)
    # Islamic investment accounts (outside deposits from 2015-07)
    ia = tbl("1.32")
    ia = ia[ia.col == 7].groupby("month").value.sum().rename("value").rename_axis("month").reset_index()
    ia["dim"], ia["category"], ia["bank"] = "investment_account", "total", "islamic"
    rows.append(ia)
    d = pd.concat(rows).dropna(subset=["value"])
    # Islamic term funding including investment accounts (IFSA 2013 moved mudharabah deposits out of deposits
    # into investment accounts by 2015-07); used for the funding-mix panel
    t = d[(d.dim == "type") & (d.bank == "islamic") & (d.category == "term")].set_index("month").value
    ia_s = ia.set_index("month").value.reindex(t.index).fillna(0.0)
    extra = (t + ia_s).rename("value").reset_index()
    extra["dim"], extra["category"], extra["bank"] = "type", "term_incl_ia", "islamic"
    d = pd.concat([d, extra])
    return d[["month", "dim", "category", "bank", "value"]]


def rates():
    cache = {t: tbl(t) for t in {t for t, _ in RATES.values()}}
    r = pd.DataFrame({k: cache[t][cache[t].col == c].groupby("month").value.mean() for k, (t, c) in RATES.items()})
    return r.rename_axis("month").reset_index()


def sme():
    rows = []
    for t, var in [("1.33", "outstanding"), ("1.33.1", "applied"), ("1.33.2", "approved"), ("1.33.3", "disbursed")]:
        L = tbl(t)
        for inst, bank in [("Islamic banks", "islamic"), ("Commercial banks", "conventional")]:
            w = collapse(L, SME_SECTOR, inst)
            long = w.rename_axis("month").reset_index().melt("month", var_name="sector", value_name="value")
            long["bank"], long["var"] = bank, var
            rows.append(long)
    s = pd.concat(rows).dropna(subset=["value"])
    return s.pivot_table(index=["month", "sector", "bank"], columns="var", values="value").reset_index()


def shocks():
    jk = pd.read_csv(RAW / "jk" / "shocks_fed_jk_m.csv")
    jk.index = pd.PeriodIndex.from_fields(year=jk["year"], month=jk["month"], freq="M")
    s = pd.DataFrame({"jk_mp": jk["MP_median"]})
    vix = pd.read_csv(RAW / "fred" / "VIXCLS.csv", na_values=".")
    vix["month"] = pd.to_datetime(vix["observation_date"]).dt.to_period("M")
    lv = np.log(vix.groupby("month")["VIXCLS"].mean())
    X = pd.concat({"l1": lv.shift(1), "l2": lv.shift(2), "l3": lv.shift(3), "mp": s["jk_mp"].reindex(lv.index)}, axis=1)
    ok = X.notna().all(axis=1)
    A = np.column_stack([np.ones(ok.sum()), X[ok].values])
    beta, *_ = np.linalg.lstsq(A, lv[ok].values, rcond=None)
    s = s.reindex(s.index.union(lv.index))
    s["vix_innov"] = pd.Series(lv[ok].values - A @ beta, index=lv[ok].index)
    d = json.loads((RAW / "bis_cbpol" / "MY.json").read_text(encoding="utf-8"))["series"]["docs"][0]
    opr = pd.Series(pd.to_numeric(pd.Series(d["value"]).replace("NA", np.nan), errors="coerce").values,
                    index=pd.PeriodIndex(d["period"], freq="M")).dropna()
    s["opr_bis"] = opr
    s["opr"] = opr_bnm()
    s["d_opr"] = s["opr"].diff()
    return s.rename_axis("month").reset_index()


def opr_bnm():
    """Period-end OPR and average overnight interbank rate, monthly block of BNM table 2.3 (year in column 1,
    month in column 2; the annual block above and the daily block below are skipped)."""
    df = pd.read_excel(RAW / "bnm" / "2.3.xlsx", header=None)
    yr, rows = None, []
    started = False
    for i in range(len(df)):
        c1, c2 = df.iloc[i, 1], df.iloc[i, 2]
        if isinstance(c1, str) and any(m in c1 for m in ["January", "February", "March", "April", "May", "June", "July",
                                                         "August", "September", "October", "November", "December"]):
            break  # daily block
        if pd.notna(c1) and str(c1).strip().split(".")[0].isdigit():
            yr = int(float(c1))
        m = pd.to_numeric(c2, errors="coerce")
        if yr and pd.notna(m) and 1 <= m <= 12:
            started = True
            rows.append((pd.Period(year=yr, month=int(m), freq="M"), pd.to_numeric(df.iloc[i, 4], errors="coerce")))
        elif started and yr is None:
            break
    return pd.Series(dict(rows)).sort_index()


def jumps(f, d, thresh=10.0):
    """Monthly |100 x dlog| above thresh in any category series, excluding the splice month (listed separately)."""
    out = []
    for frame, keys in [(f, ["dim", "category", "bank"]), (d, ["dim", "category", "bank"])]:
        for k, g in frame.groupby(keys):
            g = g.sort_values("month").set_index("month")["value"]
            g = g[g > 0]
            dl = 100 * np.log(g).diff()
            dl = dl[dl.index.to_series().diff().apply(lambda x: getattr(x, "n", 0)).eq(1)]
            big = dl[dl.abs() > thresh]
            for m, v in big.items():
                share = g.get(m)
                out.append(dict(zip(keys, k), month=str(m), dlog_pct=round(v, 1), level=round(float(share), 1),
                                at_splice=bool(m == SPLICE)))
    return pd.DataFrame(out)


def main():
    CHK.mkdir(parents=True, exist_ok=True)
    f, splice = financing()
    f.to_csv(CLEAN / "financing.csv", index=False)
    splice.to_csv(CHK / "splice_overlap.csv", index=False)
    pd.DataFrame(SPLICE_RATIOS).to_csv(CLEAN / "splice_ratio.csv", index=False)
    identity_checks().to_csv(CHK / "identity.csv", index=False)
    d = deposits()
    d.to_csv(CLEAN / "deposits.csv", index=False)
    rates().to_csv(CLEAN / "rates.csv", index=False)
    sme().to_csv(CLEAN / "sme.csv", index=False)
    shocks().to_csv(CLEAN / "shocks.csv", index=False)
    jumps(f, d).to_csv(CHK / "jumps.csv", index=False)
    neg = f[(f.bank == "conventional") & (f.value <= 0)]
    neg.to_csv(CHK / "nonpositive_conventional.csv", index=False)
    print("wrote data/clean/ and data/clean/checks/")


if __name__ == "__main__":
    main()
