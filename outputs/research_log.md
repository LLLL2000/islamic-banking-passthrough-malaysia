# Research log: islamic_malaysia_transmission

## intake_triage (2026-11-19T02:15:00Z)
- Inputs: new run after islamic_global_shocks (closed, null result at country level)
- Decision: Mode C; no dictionary match [parametric]

## model_selection, data scoping (2026-11-19T02:15:00Z)
- BNM MHS public tables: Islamic vs total-system financing by purpose/sector (2006-2026, spliced at 2021-07), deposits by type and holder (2007-), Islamic and conventional rates (2002-), investment accounts (2015-)
- Flag: financing series revised 2021-07; IFSA 2013 deposit/investment-account split (2015-06); Base Rate reform (2015-01)

## model_selection, confirmed (2026-11-19T02:40:00Z)
- Decision: H1-H4 + SMEs; shocks JK / OPR / VIX; Islamic-minus-conventional LPs; 2006-2026 [user, confirmed]
- Flag: SME tables only from 2021-07

## specification (2026-11-19T03:00:00Z)
- Decision: Stage 2 items 1-8 [user, 'continue', disclosed as interpreted]

## data_retrieval (2026-11-19T04:10:00Z)
- BNM tables (curl; urllib gets 403), OPR from BNM 2.3, JK, VIX
- Identity holds (Islamic banks row matches exactly); Islamic share 13% -> 44%
- Flags: 2021-07 splice (purpose ok, sectors reclassified), 2018-04 entry jump, Islamic rates only from 2011, primary-test count corrected to 6

## Stage 3 gate (2026-11-19T04:30:00Z)
- Decisions made by Claude under user delegation: break rules, 8 headline purposes, H1 2011-2026, 6 primary tests with Holm on permutation p

## estimation (2026-11-19T05:15:00Z)
- Revision 1: conditional placebo (OPR changes in pre-window); passes 25/27
## economic_sensibility (2026-11-19T06:00:00Z)
- Revision 2: pass-through relative to OPR's own path (Base Rate check: ~1.0 for both)
- H3 null; H1 suggestive (Holm 0.09); H2a deposit-rate stickiness strong (secondary); SME exploratory, likely confounded

## results_reporting (2026-11-19T07:30:00Z)
- Outputs: outputs/report.html; paper/working_paper.pdf
## disposition (2026-11-19T08:00:00Z)
- Decision: keep; run closed; published as a new GitHub repository [user, confirmed]
