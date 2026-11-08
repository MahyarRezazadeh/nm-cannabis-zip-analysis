"""
Quarterly border x period event study, New Mexico ZIP-code panel, per
explicit user request. Border = ZIP code within 50 miles of the nearest
incorporated place (population >= 5,000) in a state with no operating
recreational retail market -- ZIP codes near an already-legal neighbor
(Arizona, Colorado) are interior, exactly like everywhere else, not border.

Specification: log(y) ~ border_zip + border_zip:C(period) + log_population
+ C(period) + C(state_abbr), clustered by COUNTY (not ZIP), matching the
same clustering level used throughout the rest of this project's county
panels -- ZIP codes within the same county are not independent observations.

Each interaction coefficient is converted back to the period's own LEVEL
(border_zip + border_zip:C(period)[T.k], via t_test for the correct
covariance-adjusted SE), not left as a deviation from the omitted reference
period -- the same fix applied to every other event study in this project.
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
    print(f"NM ZIP-quarterly panel: {len(df)} rows, {df.zip5.nunique()} ZIP codes, "
          f"{df.county_fips.nunique()} counties, periods {df.period.min()}-{df.period.max()}")
    print(f"Border ZIP codes: {df.drop_duplicates('zip5').border_zip.sum()} of {df.zip5.nunique()}")

    specs = [
        ("Revenue per capita", "log_revenue_per_capita"),
        ("Revenue per store", "log_revenue_per_store"),
        ("Dispensary density", "log_dispensary_per_1000"),
    ]

    es_results = {}
    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_zip", "log_population", "county_fips"]).copy()
        formula = (f"{yvar} ~ border_zip + border_zip:C(period) + log_population "
                   f"+ C(period) + C(state_abbr)")
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})

        all_periods = sorted(d["period"].unique())
        interacted = {t.split("[T.")[1].rstrip("]") for t in m.params.index
                      if t.startswith("border_zip:C(period)")}
        rows = []
        for p in all_periods:
            if p in interacted:
                test_str = f"border_zip + border_zip:C(period)[T.{p}] = 0"
            else:
                test_str = "border_zip = 0"
            tt = m.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        es_results[label] = dd
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_zip_{label.replace(' ', '')}.csv"), index=False)
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
    fig.suptitle("Border x quarter event study -- New Mexico ZIP-code panel (50mi nearest-nonlegal-city "
                  "border), county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_zip.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
