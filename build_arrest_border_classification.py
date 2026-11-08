"""
Border classification for the arrest/demand-spillover discussion section:
which counties of Oklahoma, Texas, and Utah -- all still non-legal for adult-
use recreational cannabis -- sit near the state line of a neighbor that
legalized retail sales during the sample window, and when does that neighbor's
legalization date make them "treated" in a staggered design?

Three staggered cohorts, one per legalizing neighbor with a real land border
to at least one of OK/TX/UT:
  - Colorado (retail 2014-01-01): borders OK (panhandle) and UT
  - Arizona  (retail 2021-01-22): borders UT
  - New Mexico (retail 2022-04-01): borders TX and OK

Same methodology as build_county_border_stateline_variant.py elsewhere in this
project: county centroid distance (via a projected CRS, not haversine) to the
dissolved state polygon of the (single) legalizing neighbor, 75-mile threshold.
A county's treatment cohort is the EARLIEST qualifying neighbor if it happens
to be within 75 miles of more than one (e.g. a Utah county near both Colorado
and Arizona); counties never within 75 miles of any of the three are the
never-treated control group.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")

TREATMENT_STATES = ["OK", "TX", "UT"]
NEIGHBORS = {
    "CO": pd.Timestamp("2014-01-01"),
    "AZ": pd.Timestamp("2021-01-22"),
    "NM": pd.Timestamp("2022-04-01"),
}
THRESHOLD_MI = 75


def main():
    cc = pd.read_stata(os.path.join(SOURCE_BASE, "data", "county_centroids.dta"))
    cc["fips5"] = cc["fips5"].astype(str).str.zfill(5)
    counties = cc[cc.state_abbr.isin(TREATMENT_STATES)].copy()
    print(f"{len(counties)} counties across {TREATMENT_STATES}")

    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    shp_counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    shp_counties = shp_counties.to_crs(5070)
    states_poly = shp_counties.dissolve(by="state_abbr").reset_index()

    pts = gpd.GeoSeries([gpd_point(lon, lat) for lon, lat in zip(counties["lon"], counties["lat"])],
                         crs=4326).to_crs(5070)

    for nb in NEIGHBORS:
        nb_geom = states_poly.loc[states_poly.state_abbr == nb, "geometry"].values[0]
        counties[f"dist_{nb}_mi"] = pts.distance(nb_geom).to_numpy() / 1609.34

    dist_cols = [f"dist_{nb}_mi" for nb in NEIGHBORS]
    counties["nearest_neighbor"] = counties[dist_cols].idxmin(axis=1).str.replace("dist_", "").str.replace("_mi", "")
    counties["dist_nearest_legal_mi"] = counties[dist_cols].min(axis=1)

    # earliest qualifying neighbor within threshold (may differ from nearest-by-distance
    # if a county is within range of two neighbors that legalized at different times)
    def earliest_qualifying(row):
        qualifying = [(nb, NEIGHBORS[nb]) for nb in NEIGHBORS if row[f"dist_{nb}_mi"] <= THRESHOLD_MI]
        if not qualifying:
            return pd.Series({"treated_by": None, "treatment_date": pd.NaT})
        nb, date = min(qualifying, key=lambda x: x[1])
        return pd.Series({"treated_by": nb, "treatment_date": date})

    tq = counties.apply(earliest_qualifying, axis=1)
    counties = pd.concat([counties, tq], axis=1)
    counties["border_county"] = counties["treated_by"].notna().astype(int)
    counties["treatment_quarter"] = counties["treatment_date"].dt.to_period("Q").astype(str)
    counties.loc[counties["treated_by"].isna(), "treatment_quarter"] = None

    out_cols = ["fips5", "state_abbr", "county_name", "lat", "lon"] + dist_cols + \
               ["nearest_neighbor", "dist_nearest_legal_mi", "treated_by", "treatment_date",
                "treatment_quarter", "border_county"]
    out = counties[out_cols].sort_values(["state_abbr", "dist_nearest_legal_mi"])
    out_path = os.path.join(TBL, "arrest_county_border_classification.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\nBorder counties by state and treatment cohort:")
    print(out[out.border_county == 1].groupby(["state_abbr", "treated_by"]).size())
    print("\nTotals:")
    print(out.groupby("state_abbr")["border_county"].agg(["sum", "size"]))
    print("\nBorder county list:")
    print(out[out.border_county == 1][["state_abbr", "county_name", "treated_by",
                                        "dist_nearest_legal_mi", "treatment_quarter"]].to_string(index=False))


def gpd_point(lon, lat):
    from shapely.geometry import Point
    return Point(lon, lat)


if __name__ == "__main__":
    main()
