"""Figures for the working paper (paper/figures/*.png). Run after replicate.py, with MaRCo's Python."""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "outputs"
CLEAN = HERE / "data" / "clean"
FIG = HERE / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CONV, ISL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#222222", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# Figure 1: Islamic share of financing, and book rates (two separate panels, one axis each)
f = pd.read_csv(CLEAN / "financing.csv")
t = f[(f.dim == "purpose") & (f.category == "total")].pivot_table(index="month", columns="bank", values="value")
t.index = pd.PeriodIndex(t.index, freq="M").to_timestamp()
r = pd.read_csv(CLEAN / "rates.csv")
r.index = pd.PeriodIndex(r["month"], freq="M").to_timestamp()
r = r.loc["2011-01":]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(t.index, 100 * t["islamic"] / t["system"], color=ISL)
ax[0].set_title("Islamic share of banking-system financing (%)", fontsize=9, loc="left")
ax[1].plot(r.index, r["conv_alr"], color=CONV, label="Conventional ALR")
ax[1].plot(r.index, r["isl_afr"], color=ISL, label="Islamic AFR")
ax[1].set_title("Average rate on outstanding loans / financing (%)", fontsize=9, loc="left")
ax[1].legend(loc="lower left", fontsize=8)
save(fig, "fig1_share_rates.png")

# Figure 2: pass-through relative to the OPR path
p = pd.read_csv(OUT / "passthrough_ratio.csv")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for a, (name, title) in zip(ax, [("book", "Rate on outstanding financing"), ("fd12", "12-month deposit rate"),
                                  ("savings", "Savings deposit rate")]):
    a.plot(p.h, p[f"{name}_conventional"], color=CONV, marker="o", markersize=4, label="Conventional")
    a.plot(p.h, p[f"{name}_islamic"], color=ISL, marker="o", markersize=4, label="Islamic")
    a.axhline(1, color=MUTED, linewidth=0.8, linestyle="--")
    a.set_title(title, fontsize=9, loc="left")
    a.set_xlabel("Months after a +25 bp OPR change" if a is ax[1] else "")
ax[0].set_ylabel("Share of the OPR's own path")
ax[0].legend(loc="lower right", fontsize=8)
save(fig, "fig2_passthrough.png")

# Figures 3 and 4: Islamic-minus-conventional responses with 68% / 90% bands
res = pd.read_csv(OUT / "lp_results.csv")


def band(a, g, color):
    a.fill_between(g.h, g.b - 1.645 * g.se, g.b + 1.645 * g.se, color=color, alpha=0.15, linewidth=0)
    a.fill_between(g.h, g.b - g.se, g.b + g.se, color=color, alpha=0.30, linewidth=0)
    a.plot(g.h, g.b, color=color)
    a.axhline(0, color=MUTED, linewidth=0.8)
    a.set_xlabel("Months")


fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
for a, (s, title) in zip(ax, [("jk_mp", "US MP surprise (1 s.d.)"), ("d_opr", "OPR change (+25 bp)"),
                              ("vix_innov", "Global risk-off (1 s.d.)")]):
    band(a, res[(res.block == "credit") & (res.unit == "purpose") & (res.shock == s)].sort_values("h"), ISL)
    a.set_title(title, fontsize=9, loc="left")
ax[0].set_ylabel("Islamic minus conventional, %")
save(fig, "fig3_credit.png")

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
for a, (u, title) in zip(ax, [("book", "Rate on outstanding financing"), ("fd12", "12-month deposit rate"),
                              ("savings", "Savings deposit rate")]):
    band(a, res[(res.block == "pricing") & (res.unit == u) & (res.shock == "d_opr")].sort_values("h"), ISL)
    a.set_title(title, fontsize=9, loc="left")
ax[0].set_ylabel("Islamic minus conventional, pp")
save(fig, "fig4_spreads.png")
print("wrote", sorted(x.name for x in FIG.glob("*.png")))
