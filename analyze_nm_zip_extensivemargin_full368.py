"""
Extensive-margin test on the FULL 368-ZIP geography-only sample (not the
116-ZIP dispensary-activity-conditional regression sample used elsewhere):
is a border ZIP more likely to EVER record any dispensary activity at all?

This directly formalizes the over-representation noted when comparing the
full-universe border share (92/368 = 25.0%) to the regression-sample border
share (38/116 = 32.8%) -- rather than leaving it as an informal ratio
comparison, it's estimated as a proper cross-sectional regression with a
population control, and reported with a standard two-proportion test as a
simple, transparent benchmark alongside it.

Outcome: has_dispensary_zt = 1 if the ZIP appears at all in
nm_zip_panel_corrected.csv (i.e. recorded >=1 dispensary in >=1 quarter,
2022Q2-2026Q3), 0 otherwise. Cross-sectional (one observation per ZIP, not
a panel) since this is a single yes/no fact about each of the 368 ZIPs, not
a time-varying outcome.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")


def main():
    full = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    reg_sample = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"), dtype={"zip5": str})
    ever_disp_zips = set(reg_sample["zip5"].unique())

    full["has_dispensary"] = full["zip5"].isin(ever_disp_zips).astype(int)
    full["log_zip_population"] = np.log(full["zip_population"])

    print(f"Full sample: {len(full)} populated NM ZIPs")
    print(full.groupby("border_zip")["has_dispensary"].agg(["sum", "count", "mean"]))

    # ---------- simple two-proportion z-test ----------
    n_border = (full.border_zip == 1).sum()
    x_border = full.loc[full.border_zip == 1, "has_dispensary"].sum()
    n_interior = (full.border_zip == 0).sum()
    x_interior = full.loc[full.border_zip == 0, "has_dispensary"].sum()
    p_border = x_border / n_border
    p_interior = x_interior / n_interior
    p_pool = (x_border + x_interior) / (n_border + n_interior)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_border + 1 / n_interior))
    z = (p_border - p_interior) / se_pool
    p_two_sided = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"\nTwo-proportion z-test:")
    print(f"  Border: {x_border}/{n_border} = {p_border:.3f}")
    print(f"  Interior: {x_interior}/{n_interior} = {p_interior:.3f}")
    print(f"  Difference: {(p_border - p_interior)*100:+.1f} pp, "
          f"relative: {(p_border/p_interior - 1)*100:+.1f}%")
    print(f"  z={z:.3f}, p={p_two_sided:.4f}")

    # ---------- linear probability model, with population control, county-clustered ----------
    d = full.dropna(subset=["log_zip_population", "county_fips"]).copy()
    m_lpm = smf.ols("has_dispensary ~ border_zip + log_zip_population", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
    print(f"\nLinear probability model (county-clustered SEs), n={int(m_lpm.nobs)}:")
    for term in ["Intercept", "border_zip", "log_zip_population"]:
        b, se, p = m_lpm.params[term], m_lpm.bse[term], m_lpm.pvalues[term]
        print(f"  {term}: b={b:+.4f}, se={se:.4f}, p={p:.4f}")

    # ---------- logit, same controls, for robustness to functional form ----------
    m_logit = smf.logit("has_dispensary ~ border_zip + log_zip_population", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
    print(f"\nLogit (county-clustered SEs), n={int(m_logit.nobs)}, pseudo-R2={m_logit.prsquared:.3f}:")
    for term in m_logit.params.index:
        b, se, p = m_logit.params[term], m_logit.bse[term], m_logit.pvalues[term]
        print(f"  {term}: b={b:+.4f}, se={se:.4f}, p={p:.4f}")
    mfx = m_logit.get_margeff()
    print("\nAverage marginal effects (logit):")
    print(mfx.summary())

    out = pd.DataFrame([
        {"method": "two-proportion test", "border_rate": p_border, "interior_rate": p_interior,
         "diff_pp": p_border - p_interior, "p": p_two_sided, "n": len(full)},
        {"method": "LPM (pop control, county-cl.)", "border_rate": np.nan, "interior_rate": np.nan,
         "diff_pp": m_lpm.params["border_zip"], "p": m_lpm.pvalues["border_zip"], "n": int(m_lpm.nobs)},
        {"method": "Logit coef (pop control, county-cl.)", "border_rate": np.nan, "interior_rate": np.nan,
         "diff_pp": m_logit.params["border_zip"], "p": m_logit.pvalues["border_zip"], "n": int(m_logit.nobs)},
    ])
    out.to_csv(os.path.join(TBL, "nm_zip_extensivemargin_full368_results.csv"), index=False)
    print(f"\nWrote {os.path.join(TBL, 'nm_zip_extensivemargin_full368_results.csv')}")


if __name__ == "__main__":
    main()
