# Data and Background section — asset manifest

Built 2026-08-29 for the reframed Section 3 (New Mexico ZIP-level + New Mexico
dispensary-level + New Mexico vs. Colorado county-level). All files live in
**both** `New Mexico project/figures and tables/` and `Latest try/reframe/`
(figures under `reframe/figures/`, tables under `reframe/tables/`).

## Section text

| File | Purpose |
|---|---|
| `Data And Background.tex` | Self-contained replacement for Section 3. Prose + the 3 figure environments + `\input`s of the 2 tables. Compiles clean under the project preamble. Provides labels `sec:data`, `sec:samplecoverage`, `sec:sources`, `sec:summarystats`, `sec:panelstructure`, `sec:timevarying`, `map`, `fig:dispensary_map`, `fig:nm_setting`, `fig:spatial_inequality`, `tab:sumstats_nm`, `tab:sumstats_state`. |
| `data_background_new_bib_entries.bib` | 10 bibliography entries to append to `bibliography.bib`. Already present in that file: `COMarijuanaSalesByCounty`, `COMedLicensees`, `NMCCDReporting`, `CensusPopEst2023`, `PayanBrownSong2021`, `HansenMillerWeber2020`, `MTHB701`. |

## Figures

| File | Label | Goes in | Shows |
|---|---|---|---|
| `fig_nm_setting.png` | `fig:nm_setting` | §3 intro | New Mexico + neighbors shaded by recreational-legal status (CO/AZ legal, TX/OK/UT not); 92 border-ZIP centroids (red), 276 interior (blue), 1,192 dispensaries (gold). |
| `fig1_southwest_map.png` | `fig:dispensary_map` (also `map`) | §3 "Defining border and interior markets"; **replaces the old 5-state `fig_dispensary_map.png`** | Southwest-only county choropleth of dispensaries per 1,000 residents (2025). Red outlines = counties containing ≥1 border-exposed dispensary (19 NM, 17 CO). Note: the county regression uses the centroid rule (10 NM / 23 CO); the caption states both. |
| `fig_nm_spatial_inequality.png` | `fig:spatial_inequality` | §3 "Defining border and interior markets" | (a) ZIP-centroid bubble map sized by dispensaries per 1,000 residents; (b) population-weighted border vs. interior bars (1.6× density, 1.7× revenue per resident). |

## Tables

| File | Label | Goes in | Contents |
|---|---|---|---|
| `tab_sumstats_nm.tex` | `tab:sumstats_nm` | §3 "Outcome variables and summary statistics" | Panel A: ZIP-code level (368 ZIPs, 92 border / 276 interior, 6,624 ZIP-quarters). Panel B: dispensary level (1,192 dispensaries, 436 border / 756 interior, 10,739 dispensary-quarters, 684 brands). Border vs. interior means for all key variables. |
| `tab_sumstats_state.tex` | `tab:sumstats_state` | §3 "Levels of analysis and sample coverage" | New Mexico vs. Colorado county-level sample: entry regime, first retail-sales date, border-county counts (geography vs. in-sample), county-quarters, outcome means. |

## Regenerate

- Tables: `py "New Mexico project/build_data_background_tables.py"`
- Figures: `py "New Mexico project/build_data_background_figures.py"`
  (border-county classification for `fig1_southwest_map.png` is
  `dispensary_anchored_border_counties()` in that script.)

## Not yet done

- Not uploaded to Overleaf. To integrate: upload the 3 PNGs to the project root,
  delete/replace `fig_dispensary_map.png`, append the 10 bib entries, swap
  Section 3 for `Data And Background.tex`, and check that `\ref{map}` resolves to
  the new `\label{map}`.
- Table 2 / the cross-state regression still use the county-centroid border rule
  (10 NM / 23 CO). Only `fig1_southwest_map.png` uses the dispensary-anchored
  rule (19 NM / 17 CO).
