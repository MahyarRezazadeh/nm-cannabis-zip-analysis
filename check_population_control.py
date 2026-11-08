"""
Does dropping log(Population) from the baseline / dynamic border-premium
specification change the border coefficient?

The concern: for the two per-capita outcomes, y = X/P, so log(y) = log(X) - log(P),
and the RHS control gamma*log(P) puts population "on both sides." Rewriting,
  log(X) = alpha + beta*Border + (1+gamma)*log(P) + delta_t + eps,
so dropping the control forces the population elasticity of X to be exactly 1.
The question is whether that restriction moves beta (the border premium).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project"
TBL = os.path.join(BASE, "tables")

df = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"),
                 dtype={"zip5": str, "county_fips": str})
zdf = pd.read_csv(os.path.join(TBL, "nm_zip_panel_zeroinclusive.csv"),
                  dtype={"zip5": str, "county_fips": str}) if os.path.exists(
                  os.path.join(TBL, "nm_zip_panel_zeroinclusive.csv")) else None

OUT = [("log_revenue_per_capita", "Revenue per capita  (ratio -> log)"),
       ("log_dispensary_per_capita", "Dispensary density  (ratio -> log)"),
       ("log_revenue_per_dispensary", "Revenue per dispensary  (no P in outcome)")]


def fit(d, yvar, rhs):
    m = smf.ols(f"{yvar} ~ {rhs}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
    b, se, p = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
    g = m.params.get("log_population", np.nan)
    return b, se, p, g, int(m.nobs)


print("=" * 92)
print("BASELINE (eq. 1), log outcomes, county-clustered SE, 116-ZIP corrected panel")
print("=" * 92)
print(f"{'outcome':<42}{'spec':<26}{'Border b (se)':<20}{'p':<8}{'gamma_logP':<10}")
for yv, lab in OUT:
    d = df.dropna(subset=[yv, "log_population"]).copy()
    b1, se1, p1, g1, n1 = fit(d, yv, "border_zip + log_population + C(yq)")
    b0, se0, p0, g0, n0 = fit(d, yv, "border_zip + C(yq)")
    print(f"{lab:<42}{'with  log(Population)':<26}{f'{b1:+.3f} ({se1:.3f})':<20}{p1:<8.3f}{g1:<10.3f}")
    print(f"{'':<42}{'drop  log(Population)':<26}{f'{b0:+.3f} ({se0:.3f})':<20}{p0:<8.3f}{'--':<10}")
    print(f"{'':<42}{'change in Border b':<26}{b0 - b1:+.3f}")
    print("-" * 92)

# the exact algebraic equivalent: put the count/revenue in levels, P free on RHS
print("\nEquivalent check -- move population to the RHS as a level regressor:")
print("  log(total_sales) ~ Border + log(Population) + C(yq)   [per-capita spec is this with coef 1+gamma]")
d = df.dropna(subset=["total_sales", "log_population"]).copy()
d = d[d.total_sales > 0]
d["log_sales"] = np.log(d["total_sales"])
d["log_ndisp"] = np.log(d["n_dispensaries"].where(d["n_dispensaries"] > 0))
for yv, lab in [("log_sales", "log(total revenue)"), ("log_ndisp", "log(dispensary count)")]:
    dd = d.dropna(subset=[yv]).copy()
    b, se, p, g, n = fit(dd, yv, "border_zip + log_population + C(yq)")
    print(f"  {lab:<26} Border b = {b:+.3f} ({se:.3f}), p={p:.3f};  "
          f"pop elasticity (1+gamma) = {1 + g:.3f}")

if zdf is not None:
    print("\n" + "=" * 92)
    print("ZERO-INCLUSIVE asinh outcomes (368-ZIP panel, current main Table 2 cols 1-2)")
    print("=" * 92)
    for yv, lab in [("asinh_revenue_per_capita", "Revenue per capita (asinh)"),
                    ("asinh_dispensary_per_1000", "Dispensary density (asinh)")]:
        if yv not in zdf.columns:
            print(f"  ({yv} not in zero-inclusive panel; columns: {list(zdf.columns)})")
            continue
        dz = zdf.dropna(subset=[yv, "log_population"]).copy()
        b1, se1, p1, g1, n1 = fit(dz, yv, "border_zip + log_population + C(yq)")
        b0, se0, p0, g0, n0 = fit(dz, yv, "border_zip + C(yq)")
        print(f"  {lab:<32} with P: {b1:+.3f} ({se1:.3f}) p={p1:.3f}   "
              f"drop P: {b0:+.3f} ({se0:.3f}) p={p0:.3f}   dB={b0 - b1:+.3f}")
