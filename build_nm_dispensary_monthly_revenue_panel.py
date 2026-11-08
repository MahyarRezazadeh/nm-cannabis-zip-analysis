"""
Monthly dispensary-revenue panel for New Mexico, at the user's request:
the entry-game project's own dispensary_id x year x month panel
(29,895 rows: total_sales, medical_sales, adultuse_sales, plus
county/population/border covariates), merged with the geocoded
lon/lat built in build_nm_dispensary_locations_dta.py.

dispensary_id here matches nm_dispensary_locations.csv's dispensary_id
directly (both are built from this same source panel), so the location
merge is a plain key join -- no address-string matching needed this time.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")


def main():
    panel = pd.read_csv(os.path.join(SOURCE_BASE, "tables", "nm_dispensary_panel_monthly.csv"),
                         dtype={"county_fips": str, "fips5": str})
    print(f"{len(panel)} rows, {panel.dispensary_id.nunique()} dispensaries in source panel")

    loc = pd.read_csv(os.path.join(TBL, "nm_dispensary_locations.csv"))
    loc = loc[["dispensary_id", "lon", "lat", "match_status", "match_type"]]

    out = panel.merge(loc, on="dispensary_id", how="left")
    assert len(out) == len(panel), "merge changed row count -- dispensary_id isn't unique in locations file"

    out["ym"] = pd.to_datetime(out["year"].astype(str) + "-" + out["month"].astype(str) + "-01")

    n_geo = out.drop_duplicates("dispensary_id")["lat"].notna().sum()
    n_disp = out.dispensary_id.nunique()
    print(f"{n_geo} of {n_disp} dispensaries have coordinates "
          f"({n_geo/n_disp*100:.1f}%)")
    print(f"Date range: {out.ym.min().date()} to {out.ym.max().date()}")

    csv_path = os.path.join(TBL, "nm_dispensary_monthly_revenue_panel.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    dta_path = os.path.join(TBL, "nm_dispensary_monthly_revenue_panel.dta")
    out_dta = out.copy()
    str_cols = out_dta.select_dtypes(include="object").columns
    out_dta[str_cols] = out_dta[str_cols].fillna("")
    out_dta = out_dta.drop(columns=["ym"])  # Stata datetime handling differs; year/month cols already carry this
    out_dta.to_stata(dta_path, write_index=False, version=118)
    print(f"Wrote {dta_path}")


if __name__ == "__main__":
    main()
