"""
Full zero-inclusive ZIP-quarter panel for New Mexico: all 368 populated NM
ZIPs (per tables/nm_zip_border_stateline.csv) crossed with every quarter in
the existing dispensary-activity sample's own date range, so that a ZIP with
NO dispensary in a given quarter enters as a genuine zero rather than being
absent from the panel -- required to make the paper's stated design
("ZIP codes with no dispensaries remain in the sample") literally true for
revenue per capita and dispensary density.

Revenue per dispensary is NOT rebuilt here: by construction it is undefined
where n_dispensaries=0, so it correctly stays restricted to the existing
nm_zip_panel_corrected.csv sample (log, N=1,704) -- unchanged.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")


def asinh(x):
    return np.log(x + np.sqrt(x ** 2 + 1))


def main():
    border = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"), dtype={"zip5_nearest": str})

    agg = disp.groupby(["zip5_nearest", "yq"], as_index=False).agg(
        total_sales=("total_sales", "sum"),
        n_dispensaries=("dispensary_id", "nunique"),
    ).rename(columns={"zip5_nearest": "zip5"})

    quarters = sorted(agg["yq"].unique())
    print(f"Quarters in sample: {quarters[0]} to {quarters[-1]} ({len(quarters)} quarters)")
    print(f"Full ZIP universe: {border.zip5.nunique()} ZIPs")

    skeleton = border[["zip5", "border_zip", "zip_population", "county_fips", "county_name",
                        "dist_stateline_mi", "nonlegal_pop_within_75mi"]].copy()
    skeleton = skeleton.merge(pd.DataFrame({"yq": quarters}), how="cross")
    print(f"Full skeleton: {len(skeleton)} ZIP-quarter cells "
          f"({border.zip5.nunique()} ZIPs x {len(quarters)} quarters)")

    panel = skeleton.merge(agg, on=["zip5", "yq"], how="left")
    n_zero = panel["total_sales"].isna().sum()
    print(f"ZIP-quarter cells with NO dispensary activity (filled as true zero): {n_zero} of {len(panel)}")
    panel["total_sales"] = panel["total_sales"].fillna(0.0)
    panel["n_dispensaries"] = panel["n_dispensaries"].fillna(0).astype(int)

    panel["revenue_per_capita"] = panel["total_sales"] / panel["zip_population"]
    panel["dispensary_per_capita"] = panel["n_dispensaries"] / panel["zip_population"] * 1000
    panel["log_population"] = np.log(panel["zip_population"])

    panel["asinh_revenue_per_capita"] = asinh(panel["revenue_per_capita"])
    panel["asinh_dispensary_per_capita"] = asinh(panel["dispensary_per_capita"])

    out_path = os.path.join(TBL, "nm_zip_panel_zeroinclusive.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(panel)} rows, {panel.zip5.nunique()} ZIPs, {panel.yq.nunique()} quarters")
    print(f"Border ZIP-quarters: {(panel.border_zip==1).sum()}, "
          f"Interior ZIP-quarters: {(panel.border_zip==0).sum()}")
    print(f"Distinct border ZIPs: {panel.loc[panel.border_zip==1,'zip5'].nunique()}, "
          f"interior ZIPs: {panel.loc[panel.border_zip==0,'zip5'].nunique()}")
    print(f"Zero-revenue cells: {(panel.total_sales==0).sum()} of {len(panel)} "
          f"({(panel.total_sales==0).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
