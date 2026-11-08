"""
Rebuilds the New Mexico ZIP-quarter panel using the corrected, predetermined
border-treatment definition (tables/nm_zip_border_stateline.csv: ZIP
centroid distance to the nonlegal-state line, TX/OK/UT, 75-mile threshold,
>=1,000 nonlegal residents within that reach -- no dispensary or revenue
data enters the classification). Replaces the outcome-dependent
"contains a border dispensary" definition used previously.

Source dispensary-quarter data: nm_dispensary_quarterly_panel_v3.csv
(already used throughout the paper's NM analysis; unchanged here).

Outcomes (ZIP-quarter level):
  - revenue_per_capita = sum(total_sales) / zip_population
  - dispensary_per_capita = n_dispensaries / zip_population * 1000
  - revenue_per_dispensary = sum(total_sales) / n_dispensaries

Sample restrictions and dropped-observation counts are printed and logged.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")


def main():
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"),
                        dtype={"zip5_nearest": str, "fips5": str})
    n_disp_obs = len(disp)

    border = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})

    # ---------- aggregate dispensary-quarter to ZIP-quarter ----------
    agg = disp.groupby(["zip5_nearest", "yq", "year", "quarter"], as_index=False).agg(
        total_sales=("total_sales", "sum"),
        n_dispensaries=("dispensary_id", "nunique"),
    )
    agg = agg.rename(columns={"zip5_nearest": "zip5"})
    print(f"Dispensary-quarter rows: {n_disp_obs}  ->  ZIP-quarter rows after aggregation: {len(agg)}")

    # ---------- merge predetermined border classification + population ----------
    before = len(agg)
    panel = agg.merge(border[["zip5", "county_fips", "county_name", "zip_population",
                               "dist_stateline_mi", "nonlegal_pop_within_75mi", "border_zip"]],
                       on="zip5", how="left")
    missing_border = panel["border_zip"].isna()
    n_missing = missing_border.sum()
    if n_missing:
        print(f"DROPPED {n_missing} of {before} ZIP-quarter rows: ZIP not in the predetermined "
              f"border-classification file (not a populated NM ZIP in uszips_simplemaps.csv, "
              f"e.g. a PO-box-only or non-residential ZIP).")
        print(panel.loc[missing_border, "zip5"].unique())
    panel = panel.dropna(subset=["border_zip"]).copy()
    panel["border_zip"] = panel["border_zip"].astype(int)

    # ---------- outcomes ----------
    panel["revenue_per_capita"] = panel["total_sales"] / panel["zip_population"]
    panel["dispensary_per_capita"] = panel["n_dispensaries"] / panel["zip_population"] * 1000
    panel["revenue_per_dispensary"] = panel["total_sales"] / panel["n_dispensaries"]

    panel["log_revenue_per_capita"] = np.log(panel["revenue_per_capita"])
    panel["log_dispensary_per_capita"] = np.log(panel["dispensary_per_capita"])
    panel["log_revenue_per_dispensary"] = np.log(panel["revenue_per_dispensary"])
    panel["log_population"] = np.log(panel["zip_population"])
    panel["log_exposure"] = np.log1p(panel["nonlegal_pop_within_75mi"])

    out_path = os.path.join(TBL, "nm_zip_panel_corrected.csv")
    panel.to_csv(out_path, index=False)

    print(f"\nFinal panel: {len(panel)} ZIP-quarter observations, "
          f"{panel.zip5.nunique()} distinct ZIPs, {panel.yq.nunique()} quarters "
          f"({sorted(panel.yq.unique())[0]} to {sorted(panel.yq.unique())[-1]})")
    print(f"Border ZIP-quarters: {(panel.border_zip==1).sum()}, "
          f"Interior ZIP-quarters: {(panel.border_zip==0).sum()}")
    print(f"Distinct border ZIPs represented: {panel.loc[panel.border_zip==1,'zip5'].nunique()}, "
          f"interior ZIPs represented: {panel.loc[panel.border_zip==0,'zip5'].nunique()}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
