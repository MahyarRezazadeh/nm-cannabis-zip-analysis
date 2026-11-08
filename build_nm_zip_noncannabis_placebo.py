"""
ZIP-level version of the non-cannabis-sector placebo check. Unlike New
Mexico's own tax department (RP-80), which reports gross receipts at
~260 sub-county "locations" with no ZIP-code geography, the Census
Bureau's ZIP Code Business Patterns (ZBP, via the CBP API) reports
ESTABLISHMENT COUNTS by NAICS code for every ZIP code nationally,
annually, 2018-2023. This is a density-style outcome (not gross
receipts) -- directly analogous to the "dispensary density" outcome
that carries this paper's strongest and most significant cannabis
border result (49.3 percent more dispensaries per capita in border ZIP
codes), so if anything this is a MORE informative placebo than the
county-level revenue check: it asks whether border ZIP codes have more
liquor stores, grocery stores, or retailers of any kind per capita, the
same margin on which the cannabis result is strongest.

Border status uses a fresh ZIP-level 75-mile state-line distance
classification (build_nm_zip_border_stateline.py), matching this
paper's standard county-level border definition but applied to each
ZIP's own centroid, covering all 368 populated New Mexico ZIP codes
(not only the ~120 that happen to contain a dispensary).

Missing (zip, year, sector) cells are treated as zero establishments,
not suppressed: unlike New Mexico's tax data (which marks small cells
"*"), ZBP simply omits rows for zero-establishment cells rather than
suppressing nonzero small counts.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")

SECTORS = ["alcohol", "grocery", "total_retail"]
LABELS = {"alcohol": "Alcohol retail (445310)", "grocery": "Grocery stores (4451)",
          "total_retail": "Total retail trade (44-45)"}


def build_panel():
    border = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    zbp = pd.read_csv(os.path.join(TBL, "nm_zip_zbp_raw.csv"), dtype={"zip5": str})

    zips = border["zip5"].unique()
    years = sorted(zbp["year"].unique())
    full_idx = pd.MultiIndex.from_product([zips, years, SECTORS], names=["zip5", "year", "sector"])
    full = pd.DataFrame(index=full_idx).reset_index()
    full = full.merge(zbp, on=["zip5", "year", "sector"], how="left")
    full["ESTAB"] = full["ESTAB"].fillna(0)

    full = full.merge(border, on="zip5", how="left")
    full["log_estab"] = np.log1p(full["ESTAB"])
    full["log_population"] = np.log(full["zip_population"])
    return full


def run_static(panel):
    print("=== Border premium by sector, ZIP-level ZBP establishment counts ===")
    results = []
    for sector in SECTORS:
        d = panel[panel.sector == sector].dropna(subset=["log_population"]).copy()
        formula = "log_estab ~ border_zip + log_population + C(year)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
        b, se, p = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
        pct = (np.exp(b) - 1) * 100
        results.append({"sector": sector, "coef": b, "se": se, "p": p, "pct": pct,
                         "n": int(m.nobs), "n_zips": d["zip5"].nunique()})
        print(f"{sector}: b={b:+.3f} ({se:.3f}), {pct:+.1f}%, p={p:.3f}, n={int(m.nobs)}, "
              f"zips={d['zip5'].nunique()}")
    pd.DataFrame(results).to_csv(os.path.join(TBL, "nm_zip_noncannabis_placebo_results.csv"), index=False)
    return results


def run_event_study(panel, sector):
    d = panel[panel.sector == sector].dropna(subset=["log_population"]).copy()
    years = sorted(d["year"].unique())
    ref_y = years[0]

    formula = f"log_estab ~ border_zip*C(year, Treatment(reference={ref_y})) + log_population"
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})

    rows = []
    for y in years:
        if y == ref_y:
            b, se, p = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
        else:
            term = f"border_zip:C(year, Treatment(reference={ref_y}))[T.{y}]"
            if term not in m.params.index:
                continue
            tt = m.t_test(f"border_zip + {term}")
            b = float(np.ravel(tt.effect)[0])
            se = float(np.ravel(tt.sd)[0])
            p = float(np.ravel(tt.pvalue)[0])
        rows.append({"sector": sector, "year": y, "coef": b, "se": se, "p": p})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(TBL, f"nm_zip_noncannabis_placebo_eventstudy_{sector}.csv"), index=False)
    n_sig = (ev["p"] < 0.05).sum()
    print(f"{sector}: {n_sig}/{len(ev)} years significant at p<0.05, "
          f"coef range [{ev['coef'].min():+.2f}, {ev['coef'].max():+.2f}]")
    return ev


LEGALIZATION_YEAR = 2022  # NM retail cannabis launched 2022-04-01 (2022Q2)


def run_legalization_event_study(panel, sector):
    """
    Matches this paper's own equation (1) structure exactly, applied to a
    non-cannabis outcome: y_zt = alpha + Border_z + sum_t beta_t*(Border_z x
    1[period=t]) + gamma*log(Population_z) + delta_t + eps_zt, reference
    period t=0 = New Mexico's own cannabis legalization year (2022), SEs
    clustered on ZIP code (not county). Since alcohol/grocery/retail legality
    never changes, this asks a sharper question than a generic calendar-time
    check: does the border GAP in these outcomes show any discontinuity
    right at the same date cannabis legalized, the way the cannabis border
    premium itself does not exist before 2022Q2 by construction?
    """
    d = panel[panel.sector == sector].dropna(subset=["log_population"]).copy()
    d["rel_year"] = d["year"] - LEGALIZATION_YEAR

    formula = "log_estab ~ border_zip*C(rel_year, Treatment(reference=0)) + log_population"
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["zip5"]})

    rel_years = sorted(d["rel_year"].unique())
    rows = []
    for ry in rel_years:
        if ry == 0:
            b, se, p = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
        else:
            term = f"border_zip:C(rel_year, Treatment(reference=0))[T.{ry}]"
            if term not in m.params.index:
                continue
            tt = m.t_test(f"border_zip + {term}")
            b = float(np.ravel(tt.effect)[0])
            se = float(np.ravel(tt.sd)[0])
            p = float(np.ravel(tt.pvalue)[0])
        rows.append({"sector": sector, "rel_year": ry, "year": ry + LEGALIZATION_YEAR,
                     "coef": b, "se": se, "p": p})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(TBL, f"nm_zip_noncannabis_placebo_legaleventstudy_{sector}.csv"), index=False)
    n_sig = (ev["p"] < 0.05).sum()
    print(f"{sector}: {n_sig}/{len(ev)} relative-years significant at p<0.05, "
          f"coef range [{ev['coef'].min():+.2f}, {ev['coef'].max():+.2f}]")
    return ev, m


def main():
    panel = build_panel()
    panel.to_csv(os.path.join(TBL, "nm_zip_noncannabis_placebo_panel.csv"), index=False)
    print(f"Wrote panel, {len(panel)} rows, {panel.zip5.nunique()} ZIPs, {panel.year.nunique()} years\n")

    run_static(panel)

    print("\n=== Calendar-year event study ===")
    ev_all = {}
    for sector in SECTORS:
        ev_all[sector] = run_event_study(panel, sector)

    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, sector in zip(axes, SECTORS):
        ev = ev_all[sector]
        xt = np.arange(len(ev))
        ax.errorbar(xt, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=3,
                    markersize=4, linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(ev["year"], rotation=45, fontsize=8)
        ax.set_title(LABELS[sector], fontsize=10)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("Border coefficient (log points)")
    fig.suptitle("ZIP-level border coefficient by year, non-cannabis establishment counts, New Mexico\n"
                 "County-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(OUT, "fig_nm_zip_noncannabis_placebo_eventstudy.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    # ---------- event study anchored to NM's own legalization date, ZIP-clustered ----------
    print("\n=== Event study anchored to legalization date (2022), ZIP-clustered SEs ===")
    leg_ev_all = {}
    for sector in SECTORS:
        ev, m = run_legalization_event_study(panel, sector)
        leg_ev_all[sector] = ev
        print(f"  static border_zip coef at rel_year=0 (2022): "
              f"{ev.loc[ev.rel_year == 0, 'coef'].values[0]:+.3f}, "
              f"p={ev.loc[ev.rel_year == 0, 'p'].values[0]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, sector in zip(axes, SECTORS):
        ev = leg_ev_all[sector]
        ax.errorbar(ev["rel_year"], ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=3,
                    markersize=4, linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(-0.5, color="red", linewidth=1.0, linestyle=":")
        ax.set_title(LABELS[sector], fontsize=10)
        ax.set_xlabel("Years relative to NM legalization (2022)")
    axes[0].set_ylabel("Border coefficient (log points)")
    fig.suptitle("ZIP-level border coefficient around New Mexico's cannabis legalization date,\n"
                 "non-cannabis establishment counts, ZIP-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    fig_path2 = os.path.join(OUT, "fig_nm_zip_noncannabis_placebo_legaleventstudy.png")
    fig.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {fig_path2}")


if __name__ == "__main__":
    main()
