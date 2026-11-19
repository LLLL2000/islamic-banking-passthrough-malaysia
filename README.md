# Same Shock, Different Prices: Islamic and Conventional Banks in Malaysia's Monetary Transmission

Working paper and replication package. **Paper:** [`paper/working_paper.pdf`](paper/working_paper.pdf)
(source: [`paper/working_paper.md`](paper/working_paper.md)).

## Question and answer

Do Islamic banks transmit monetary shocks differently from conventional banks? The paper compares the two inside
Malaysia, in the same month and for the same loan purpose, using public monthly Bank Negara Malaysia (BNM)
statistics for 2006-2026. Shocks: US monetary policy surprises (Jarociński-Karadi), changes in the overnight policy
rate (OPR) and global risk-off (VIX) shocks.

- **Credit quantities: no difference.** Islamic and conventional financing for the same purpose respond alike to all
  three shocks (primary-test permutation p = 0.83 and 0.61); the null holds in every robustness check.
- **Prices: Islamic banks reprice more slowly.** Relative to the OPR's own path, the rate on outstanding Islamic
  financing passes through 0.83 after 12 months against 0.98 for conventional loans (Base Rates: about 1.0 for both).
  Islamic 12-month deposit profit rates move 0.28 on impact against 0.89 and catch up after about 18 months; Islamic
  savings rates pass through about half as much.
- The book-rate gap is suggestive (Holm-adjusted p = 0.09) and leans on the 2022-23 hiking cycle; the deposit-rate gap
  is larger and more robust.

## Replication

Python 3.11 with numpy, pandas, statsmodels, matplotlib, openpyxl and pyyaml. From the repository root:

```bash
python -I fetch_data.py      # downloads to data/raw/ (BNM via curl), manifest with SHA-256
python -I build_panel.py     # data/clean/ panels and data/clean/checks/
python replicate.py          # baseline local projections, placebo, primary tests, pass-through ratios
python stage5.py             # robustness
python make_figures.py       # paper/figures/
python -I paper/build_pdf.py # paper/working_paper.pdf (headless Microsoft Edge)
```

`config.yaml` holds every setting the estimation uses. `make_report.py` builds the interactive
[`outputs/report.html`](outputs/report.html) with the MaRCo@BIS framework, which is not included here.

| File | Role |
|---|---|
| `fetch_data.py` | Downloads BNM tables, OPR, JK shocks and VIX unchanged; writes `data/raw/manifest.csv` |
| `bnm_parse.py` | Reads BNM's bilingual XLSX layout into long format |
| `build_panel.py` | Harmonises purposes/sectors across the 2021 revision; conventional = system minus Islamic; checks |
| `replicate.py` | Credit panel, price and funding spreads, SME block; permutation inference; Holm on primary tests |
| `stage5.py` | Robustness: 6 lags, without 2020-21, without 2022-23, value-weighted credit |
| `outputs/` | Results (CSV), `report.html`, and the research log (`session_research_log.json`, `research_log.md`) |

The research log records every design decision, who made it and why, including the six primary tests fixed before
estimation and the two revisions made after the first estimates.

## Data

All data are public. BNM *Monthly Highlights and Statistics* tables 1.19, 1.19a, 1.19.1, 1.19.1a, 1.20, 1.20a, 1.24,
1.24.2, 1.25, 1.32, 1.33, 1.33.1-1.33.3, 1.34, 2.1, 2.2 and 2.3 (source: Bank Negara Malaysia); US monetary policy
surprises from Jarociński and Karadi's public update; VIX from FRED. BNM returns HTTP 403 to Python's default HTTP
client and serves `curl`, so BNM files are fetched with `curl` under its default identity.
