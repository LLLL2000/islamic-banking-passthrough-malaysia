"""Stage 6 report for islamic_malaysia_transmission: outputs/report.html, built with MaRCo's VARReport.

Run after replicate.py and stage5.py, with MaRCo's Python:  <MaRCo>/.venv/Scripts/python.exe make_report.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, r"C:\Users\consu\Documents\Universidad\Tesis\extension\tools\marco-BIS\scripts")
import lib  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "outputs"
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
res = pd.read_csv(OUT / "lp_results.csv")
pre = pd.read_csv(OUT / "lp_placebo.csv")
prim = pd.read_csv(OUT / "primary_tests.csv")
rob = pd.read_csv(OUT / "stage5_robustness.csv")
ptr = pd.read_csv(OUT / "passthrough_ratio.csv")
diag = json.loads((OUT / "diagnostics.json").read_text(encoding="utf-8"))
log = json.loads((OUT / "session_research_log.json").read_text(encoding="utf-8"))
SH = {"jk_mp": "US monetary policy surprise (1 s.d.)", "d_opr": "OPR change (+25 bp)", "vix_innov": "Global risk-off (VIX, 1 s.d.)"}
SHOCKS = list(SH)
H = CFG["horizons"]["max"]


def table(df):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f'<td class="num">{v}</td>' if isinstance(v, (int, float, np.floating)) else f"<td>{v}</td>"
                                     for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><tr>{head}</tr>{body}</table>"


def paras(text):
    return '<div class="narrative">' + "".join(f"<p>{p.strip()}</p>" for p in text.strip().split("\n\n")) + "</div>"


def irf_block(block, units, labels, title, units_label):
    n = len(units)
    pt, l68, h68, l90, h90 = (np.full((H + 1, n, len(SHOCKS)), np.nan) for _ in range(5))
    for i, u in enumerate(units):
        for j, s in enumerate(SHOCKS):
            g = res[(res.block == block) & (res.unit == u) & (res.shock == s)].set_index("h").reindex(range(H + 1))
            pt[:, i, j] = g.b
            l68[:, i, j], h68[:, i, j] = g.b - g.se, g.b + g.se
            l90[:, i, j], h90[:, i, j] = g.b - 1.645 * g.se, g.b + 1.645 * g.se
    rep.add_irf(pt, l68, h68, l90, h90, var_names=labels, shock_labels=[SH[s] for s in SHOCKS],
                x_axis_label="Months", units=units_label, section_title=title)


rep = lib.VARReport("Islamic and conventional banks under the same shock: Malaysia, 2006-2026",
                    subtitle="Within-country local projections on Bank Negara Malaysia statistics. Working paper draft.")

rep.sections.append(("Summary", paras("""
Question. When the same shock hits Malaysia, do Islamic banks transmit it differently from conventional banks?
The comparison is within the country, month and loan purpose, so everything common to Malaysia cancels.

Credit quantities: no difference. Islamic and conventional financing for the same purpose respond alike to US
monetary policy surprises, changes in the overnight policy rate (OPR) and global risk-off shocks. Both primary
credit tests are far from significance (permutation p = 0.83 and 0.61) and the null holds in every robustness check.

Prices: Islamic banks reprice more slowly on both sides of the balance sheet. Measured against the OPR's own
cumulative path, the average rate on outstanding Islamic financing passes through 0.83 at 12 months against 0.98 for
conventional loans, while both banks' Base Rates pass through about 1.0. The Islamic 12-month deposit profit rate
moves 0.28 on impact against 0.89 and catches up after about 18 months; Islamic savings rates pass through about half
as much. The book-rate result is suggestive (permutation p = 0.015 and 0.037; Holm-adjusted 0.09 and 0.15) and is
concentrated in the 2022-23 hiking cycle; the deposit-rate result is the strongest in the run but was not a
pre-registered primary test.

Reading. Islamic banking in Malaysia smooths the price side of monetary transmission without changing the volume of
credit. That is consistent with fixed-price sale contracts in the financing book and smoothed profit rates on the
funding side, and it explains why a cross-country test on money-market rates and credit totals finds nothing.
""")))

rep.add_specification({
    "Method": "Local projections, horizons 0-24 months; Islamic minus conventional",
    "Credit panel": "Purpose x {Islamic, conventional}; purpose x month and purpose x bank fixed effects; 3 own lags; 8 purposes",
    "Price and funding blocks": "Time series of the Islamic-minus-conventional spread or funding share; 3 lags of the spread change and of the OPR change; 2015 step dummies",
    "Conventional": "Whole banking system minus Islamic banking system (BNM tables 1.19/1.20 and 1.19.1)",
    "Samples": "Financing 2006-12 to 2026-09; deposits 2007-01 on; rates 2011-01 on; SMEs 2021-07 on",
    "Shocks": "; ".join(SH.values()),
    "Inference": "Driscoll-Kraay (panel) / Newey-West (time series), lags h+2; 12-month block permutation, 999 draws",
    "Primary tests": "Six, fixed before estimation; Holm correction on permutation p-values",
})

p = prim.copy()
p["test"] = p.apply(lambda r: f'{"Credit (H3)" if r.block == "credit" else "Book rate spread (H1)"}, {SH[r.shock]}, h={r.h}', axis=1)
rep.sections.append(("Primary tests (fixed before estimation)",
                     table(p[["test", "b", "se", "p_perm", "p_holm"]].round(3)) +
                     paras("Credit: percent, Islamic minus conventional. Book-rate spread: percentage points.")))

pt_cols = ["h", "base_conventional", "base_islamic", "book_conventional", "book_islamic", "fd12_conventional",
           "fd12_islamic", "savings_conventional", "savings_islamic", "ia12_islamic"]
rep.sections.append(("Pass-through relative to the OPR's own path", table(ptr[pt_cols].round(2)) + paras(
    "Each cell is a rate's cumulative response to a +25 bp OPR change divided by the OPR's own cumulative response at "
    "the same horizon. BNM moves in sequences, so one move predicts further moves; the ratio is the economically "
    "meaningful pass-through. The Base Rate is set mechanically from the OPR and serves as a check.")))

irf_block("credit", ["purpose", "sector"], ["Financing by purpose (%)", "Financing by sector (%)"],
          "Credit: Islamic minus conventional financing growth", "Percent")
irf_block("pricing", ["book", "fd1", "fd12", "savings"],
          ["Book rate on outstanding (pp)", "1-month deposit (pp)", "12-month deposit (pp)", "Savings (pp)"],
          "Prices: Islamic minus conventional rate spread", "pp")
irf_block("funding", ["term", "demand", "savings"], ["Term funding share (pp)", "Demand deposit share (pp)", "Savings share (pp)"],
          "Funding mix: Islamic minus conventional share of funding", "pp")

# robustness
keep = rob[rob.unit.isin(["book", "fd12", "savings", "purpose"]) & rob.shock.isin(["d_opr", "jk_mp"]) & (rob.block != "passthrough")]
base = res[res.block.isin(["credit", "pricing"]) & res.unit.isin(["book", "fd12", "savings", "purpose"]) & res.h.isin([6, 12, 24]) & res.shock.isin(["d_opr", "jk_mp"])].assign(check="baseline")
allr = pd.concat([base, keep])
allr["cell"] = allr.apply(lambda r: f"{r.b:.3f} ({r.b / r.se:.1f})", axis=1)
wide = allr.pivot_table(index=["block", "unit", "shock", "h"], columns="check", values="cell", aggfunc="first").reset_index().fillna("-")
rep.sections.append(("Robustness", table(wide) + paras(
    "Coefficient (t-statistic). no_covid drops windows touching 2020-03..2021-12 (loan repayment moratorium); "
    "no_2022_23 drops windows touching 2022-05..2023-12; lags6 uses six own lags; value_weight weights purposes by their "
    "lagged system share.")))

pl = pre.copy()
pl["placebo"] = pl["placebo"].fillna("unconditional")
pl["t"] = (pl.b / pl.se).round(2)
plw = pl.pivot_table(index=["block", "unit"], columns=["placebo", "shock"], values="t").round(2).reset_index()
plw.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in plw.columns]
dg = pd.DataFrame([
    {"Check": "Shock AR(1): JK / OPR change / VIX", "Value": " / ".join(f"{diag['shock_ar1'][s]:.2f}" for s in SHOCKS)},
    {"Check": "corr(JK, OPR change)", "Value": f"{diag['corr']['jk_mp']['d_opr']:.2f}"},
    {"Check": "OPR moves in estimation sample", "Value": diag["opr_moves"]},
    {"Check": "1 s.d. JK surprise", "Value": f"{100 * diag['shock_sd']['jk_mp_sd_pp']:.1f} bp"},
])
rep.sections.append(("Diagnostics", table(dg) + "<h3>Placebo t-statistics: y(t-1) - y(t-5) on the shock at t</h3>" + table(plw) + paras(
    "The conditional placebo controls for the OPR changes inside the pre-window, matching the information the baseline "
    "conditions on. The unconditional version fails for deposit spreads because BNM moves in sequences.")))

src = pd.DataFrame([
    ("Financing by purpose and sector", "BNM Monthly Highlights & Statistics 1.19.1a/1.19.1 (Islamic), 1.19a/1.19, 1.20a/1.20 (system)"),
    ("Deposits", "1.24.2 (Islamic, by type and holder), 1.24 and 1.25 (system); investment accounts 1.32"),
    ("Rates", "2.1 (commercial banks), 2.2 (Islamic banks and Islamic windows), 2.3 (OPR, period end)"),
    ("SMEs", "1.33, 1.33.1-1.33.3 (Islamic banks vs commercial banks)"),
    ("Shocks", "Jarocinski-Karadi US surprises (authors' GitHub); VIX (FRED); OPR (BNM 2.3, checked against BIS)"),
    ("Access note", "BNM refuses Python's urllib (HTTP 403) and serves curl; files fetched with curl's default identity, no spoofing"),
], columns=["Data", "Source"])
rep.sections.append(("Data", table(src)))

rep.add_narrative("""
Search disclosure. The six primary tests and the Holm correction were fixed before estimation. Two revisions followed
the first estimates. (1) A conditional placebo was added because the baseline conditions on lagged OPR changes and the
unconditional placebo does not. (2) Pass-through is reported relative to the OPR's own path, because the raw responses
implied that the Base Rate moves more than twice the OPR. Neither revision changes the primary tests. Intervals do not
account for these revisions or for the many secondary horizons shown.

Limits. Only 15 OPR moves fall in the 2011-2026 rate sample, five of them in 2022-23; without that cycle the book-rate
gap halves and its permutation p-value rises to about 0.2. The book-rate conditional placebo is borderline
(t = -1.89). The SME block covers 2021-07 to 2026-09 only and finds implausibly large Islamic SME growth after hikes,
most likely the sector's trend coinciding with the 2022-23 hikes; it is descriptive. Conventional series are derived
by subtraction and inherit a 0.6% gap in the pre-2021 system totals.

Interpretation. Both price gaps point the same way: Islamic banks adjust what they charge and what they pay more
slowly. On the asset side, sale-based contracts (murabaha, bai' bithaman ajil) fix the price at inception, so the
outstanding book reprices only as contracts run off. On the funding side, profit rates are managed against reserves
and investment-account returns follow realised asset returns. Quantities do not differ: borrowers and depositors face
slightly different price paths, but the volume of credit in each purpose responds the same.
""")
rep.add_research_log(log)
rep.save(OUT / "report.html")
print("wrote outputs/report.html")
