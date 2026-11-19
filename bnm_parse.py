"""Parse Bank Negara Malaysia 'Monthly Highlights & Statistics' XLSX tables into long format.

Layout of these sheets: a block of bilingual (Malay / English) header rows, then data rows with the year in
column 0 (only on the first month of each year), the month in column 1 and, in bank-type tables, the institution
in column 2. Raw files are read as data only.
"""
import re

import numpy as np
import pandas as pd

YEAR = re.compile(r"^(19|20)\d\d(\.0)?$")


def english(cell):
    """Last non-empty line of a bilingual cell (BNM puts the English text last), footnote digits stripped."""
    s = str(cell)
    if s == "nan":
        return ""
    parts = [p.strip() for p in re.split(r"\n+|\s/\s", s) if p.strip()]
    if not parts:
        return ""
    return re.sub(r"(?<=[A-Za-z\)])\d+$", "", parts[-1]).strip()


def read_table(path):
    """Return (long DataFrame, column labels). Long columns: month (Period), inst, col, label, value."""
    df = pd.read_excel(path, header=None)
    first = next(i for i in range(len(df)) if YEAR.match(str(df.iloc[i, 0]).strip()))
    hdr = df.iloc[:first].copy()
    hdr = hdr.ffill(axis=1)  # merged header cells
    labels = {}
    for j in range(df.shape[1]):
        path_ = [english(v) for v in hdr.iloc[:, j]]
        path_ = [p for p in dict.fromkeys(path_) if p and p not in ("RM million", "Percent per annum", "Period")]
        labels[j] = " | ".join(path_)
    body = df.iloc[first:].copy()
    year = pd.to_numeric(body[0], errors="coerce").where(body[0].astype(str).str.strip().str.match(YEAR)).ffill()
    month = pd.to_numeric(body[1], errors="coerce")
    has_inst = body[2].map(lambda v: isinstance(v, str)).mean() > 0.3
    if has_inst:
        month = month.ffill()
        inst = body[2].map(english)
        vcols = list(range(3, df.shape[1]))
    else:
        inst = pd.Series("", index=body.index)
        vcols = list(range(2, df.shape[1]))
    ok = year.notna() & month.between(1, 12) & ((inst != "") | (not has_inst))
    body, year, month, inst = body[ok], year[ok], month[ok], inst[ok]
    per = pd.PeriodIndex.from_fields(year=year.astype(int).values, month=month.astype(int).values, freq="M")
    out = []
    for j in vcols:
        v = pd.to_numeric(body[j], errors="coerce")
        out.append(pd.DataFrame({"month": per, "inst": inst.values, "col": j, "label": labels[j], "value": v.values}))
    long = pd.concat(out, ignore_index=True).dropna(subset=["value"])
    return long, labels
