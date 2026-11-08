"""
Baseline border-premium regressions on the zero-inclusive 368-ZIP panel,
matching the paper's stated design: revenue per capita and dispensary
density retain ZIP-quarters with zero dispensary activity (asinh, since
log(0) is undefined); revenue per dispensary stays restricted to ZIP-quarters
with >=1 dispensary (log, unchanged from nm_zip_panel_corrected.csv) since
it is undefined otherwise, exactly as the text states.

Same estimating equation as before (border_zip + log_population + C(yq)),
county-clustered SEs.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")


def run(d, yvar, label, cluster_col="county_fips"):
    dd = d.dropna(subset=[yvar, "log_population"]).copy()
    m = smf.ols(f"{yvar} ~ border_zip + log_population + C(yq)", data=dd).fit(
        cov_type="cluster", cov_kwds={"groups": dd[cluster_col]})
    row = {"outcome": label,
           "intercept": m.params["Intercept"], "intercept_se": m.bse["Intercept"], "intercept_p": m.pvalues["Intercept"],
           "border_coef": m.params["border_zip"], "border_se": m.bse["border_zip"], "border_p": m.pvalues["border_zip"],
           "pop_coef": m.params["log_population"], "pop_se": m.bse["log_population"], "pop_p": m.pvalues["log_population"],
           "r2": m.rsquared, "n": int(m.nobs), "n_zips": dd["zip5"].nunique()}
    print(f"\n=== {label} ===")
    for k in ["Intercept", "border_zip", "log_population"]:
        print(f"  {k}: b={m.params[k]:+.4f}, se={m.bse[k]:.4f}, p={m.pvalues[k]:.4f}")
    print(f"  R2={m.rsquared:.4f}, n={int(m.nobs)}, ZIPs={dd['zip5'].nunique()}")
    return row


def main():
    zero_incl = pd.read_csv(os.path.join(TBL, "nm_zip_panel_zeroinclusive.csv"), dtype={"zip5": str, "county_fips": str})
    revperdisp = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"), dtype={"zip5": str, "county_fips": str})

    rows = []
    rows.append(run(zero_incl, "asinh_revenue_per_capita", "Revenue per capita (asinh, zero-inclusive)"))
    rows.append(run(zero_incl, "asinh_dispensary_per_capita", "Dispensary density (asinh, zero-inclusive)"))
    rows.append(run(revperdisp, "log_revenue_per_dispensary", "Revenue per dispensary (log, >=1 dispensary only, unchanged)"))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TBL, "nm_zip_baseline_zeroinclusive_results.csv"), index=False)
    print(f"\nWrote {os.path.join(TBL, 'nm_zip_baseline_zeroinclusive_results.csv')}")


if __name__ == "__main__":
    main()
