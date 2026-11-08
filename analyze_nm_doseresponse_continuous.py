"""
Continuous (linear) distance companion to analyze_nm_doseresponse.py's
25-mile binned specification: log(y) ~ dist_stateline_mi + log(Population) +
C(period), county-clustered SEs, same three outcomes and same panels
(nm_dispensary_doseresponse_panel.csv / nm_zip_doseresponse_panel.csv), same
175-200mi Bernalillo-driven exclusion for consistency with the binned run.

Coefficient is per-mile; also reported per-25-miles (dividing distance by 25
before regressing) so it's directly comparable in scale to one bin-step in
the binned table.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")

MAX_DIST = 175  # matches the binned analysis's 175-200mi exclusion


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def run_spec(df, yvar, popvar, cluster_var, label):
    d = df.dropna(subset=[yvar, "dist_stateline_mi", popvar, cluster_var]).copy()
    d = d[d["dist_stateline_mi"] <= MAX_DIST].copy()
    d["dist_per25mi"] = d["dist_stateline_mi"] / 25.0
    m = smf.ols(f"{yvar} ~ dist_per25mi + {popvar} + C(period)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster_var]})
    b, se, p = m.params["dist_per25mi"], m.bse["dist_per25mi"], m.pvalues["dist_per25mi"]
    n = int(m.nobs)
    print(f"[{label}] continuous, per 25mi: b={b:+.3f} ({se:.3f}), {fmt_pct(b)}, p={p:.4f}, n={n}")
    return {"outcome": label, "coef_per25mi": b, "se_per25mi": se, "p": p, "n": n}


def main():
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_doseresponse_panel.csv"), dtype={"fips5": str})
    zipp = pd.read_csv(os.path.join(TBL, "nm_zip_doseresponse_panel.csv"), dtype={"fips5": str})

    specs = [
        ("Revenue per capita", disp, "log_revenue_percapita", "log_population", "fips5"),
        ("Revenue per dispensary", disp, "log_revenue", "log_population", "fips5"),
        ("Dispensary density", zipp, "log_density_zip", "log_zip_population", "fips5"),
    ]
    rows = [run_spec(df, yvar, popvar, cluster_var, label) for label, df, yvar, popvar, cluster_var in specs]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "doseresponse_nm_continuous.csv"), index=False)
    print(f"\nWrote {os.path.join(OUT, 'doseresponse_nm_continuous.csv')}")


if __name__ == "__main__":
    main()
