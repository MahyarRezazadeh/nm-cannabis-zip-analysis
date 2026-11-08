"""
Robustness check on the main NM ZIP-code event study (border_zip vs.
interior, 50mi nearest-nonlegal-city definition), adding county fixed
effects, per explicit user request. State x year FE is not added as a
SEPARATE term: this is an NM-only panel (state is a single level, adds
nothing) already using full year-quarter period FE, which strictly nests
year FE -- adding a coarser C(year) on top of the finer C(period) already
in the model would just be redundant/collinear with it, not a new control.
County FE is the genuinely new addition here.

Feasibility checked directly (not assumed) via patsy design-matrix rank:
border_zip + C(county_fips) is FULL RANK for this design (50 of 50 columns)
-- 4 of NM's 33 counties (Curry, Doña Ana, Lea, Otero) contain BOTH
border and interior ZIP codes, so border_zip is identified off within-county
variation in those 4 counties once county FE is added. (This is NOT true
for the separate OK/TX-vs-CO/AZ placebo check -- there, treatment and
control ZIP codes fall in completely disjoint counties, so county FE would
make that specific design's treatment term exactly collinear/unidentified;
county FE is not run for that placebo check for this reason.)

Specification: log(y) ~ border_zip + border_zip:C(period) + log_population
+ C(period) + C(county_fips), clustered by county.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(TBL, "nm_zip_quarterly_panel.csv"), dtype={"zip5": str, "county_fips": str})
    df["period"] = df["yq"]

    mixed_counties = (df.drop_duplicates("zip5").groupby("county_fips")["border_zip"]
                       .nunique().pipe(lambda s: s[s > 1]).index.tolist())
    print(f"NM ZIP-quarterly panel: {len(df)} rows, {df.zip5.nunique()} ZIP codes, "
          f"{df.county_fips.nunique()} counties")
    print(f"Counties with BOTH border and interior ZIPs (identify border_zip within county FE): "
          f"{mixed_counties}")

    specs = [
        ("Revenue per capita", "log_revenue_per_capita"),
        ("Revenue per store", "log_revenue_per_store"),
        ("Dispensary density", "log_dispensary_per_1000"),
    ]

    es_results = {}
    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_zip", "log_population", "county_fips"]).copy()
        formula = f"{yvar} ~ border_zip + border_zip:C(period) + log_population + C(period) + C(county_fips)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})

        all_periods = sorted(d["period"].unique())
        interacted = {t.split("[T.")[1].rstrip("]") for t in m.params.index
                      if t.startswith("border_zip:C(period)")}
        rows = []
        for p in all_periods:
            test_str = (f"border_zip + border_zip:C(period)[T.{p}] = 0" if p in interacted
                        else "border_zip = 0")
            tt = m.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        es_results[label] = dd
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_zip_countyFE_{label.replace(' ', '')}.csv"),
                   index=False)
        print(f"{label}: n={len(d)}, {len(all_periods)} quarters, "
              f"border main (reference-period) effect b={m.params.get('border_zip', float('nan')):+.3f} "
              f"p={m.pvalues.get('border_zip', float('nan')):.3f}")

    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (label, dd) in zip(axes, es_results.items()):
        xt = np.arange(len(dd))
        ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                    linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(dd["period"], rotation=90, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Border coefficient (log points)")
    fig.suptitle("Border x quarter event study, NM ZIP panel, WITH county fixed effects\n"
                  "(identified off 4 counties with within-county border/interior variation), "
                  "county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_zip_countyFE.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
