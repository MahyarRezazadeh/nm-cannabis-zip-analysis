"""
Parameterized version of analyze_county_border_capped_main_100mi.py --
same specification, arbitrary distance threshold (uses whichever
county_border_stateline_{N}mi.csv file exists). Sample truncated to
<=2025Q2 for all states, matching the current state of the 75mi/100mi
scripts (see analyze_county_border_capped_main.py).
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def fmt_pct_plain(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def load(threshold_mi):
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    border = pd.read_csv(os.path.join(TBL, f"county_border_stateline_{threshold_mi}mi.csv"), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county"]], on="fips5", how="left")
    df["is_capped"] = df["entry_capped"]
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    print(f"{len(df)} county-quarter rows, {df.fips5.nunique()} counties, "
          f"{df.drop_duplicates('fips5').border_county.sum()} border counties (threshold={threshold_mi}mi, "
          f"sample truncated to <=2025Q2)")
    return df


def main_table(df, threshold_mi):
    specs = ["log_revenue_per_capita", "log_revenue_per_store", "log_dispensary_per_1000"]
    labels = ["Revenue per capita", "Revenue per store", "Dispensary density"]
    for label, yvar in zip(labels, specs):
        d = df.dropna(subset=[yvar, "border_county", "is_capped"]).copy()
        formula = f"{yvar} ~ border_county + border_county:is_capped + C(state_period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b_b, p_b = m.params.get("border_county", np.nan), m.pvalues.get("border_county", np.nan)
        b_i, p_i = m.params.get("border_county:is_capped", np.nan), m.pvalues.get("border_county:is_capped", np.nan)
        n = int(m.nobs)
        n_border = d.drop_duplicates("fips5").border_county.sum()
        print(f"[{threshold_mi}mi] {label}: n={n}, border counties={n_border}, "
              f"border={fmt_pct_plain(b_b)} (b={b_b:+.3f}, p={p_b:.3f}), "
              f"border x capped={fmt_pct_plain(b_i)} (b={b_i:+.3f}, p={p_i:.3f})")


if __name__ == "__main__":
    threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    df = load(threshold)
    main_table(df, threshold)
