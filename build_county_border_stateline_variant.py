"""
Parameterized county-level border classification, same methodology as
build_county_border_stateline.py, for an arbitrary distance threshold
(used here to build the 70-mile version alongside the existing 75mi/
100mi files).
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
MIN_NONLEGAL_POP = 1000
EARTH_RADIUS_MI = 3958.8
SAMPLE_STATES = ["CO", "OR", "MT", "NM"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def main(threshold_mi):
    cc = pd.read_stata(os.path.join(SOURCE_BASE, "data", "county_centroids.dta"))
    cc["fips5"] = cc["fips5"].astype(str).str.zfill(5)
    counties = cc[cc.state_abbr.isin(SAMPLE_STATES)].copy()
    print(f"{len(counties)} counties across {SAMPLE_STATES}")

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    shp_counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    shp_counties = shp_counties.to_crs(5070)
    states_poly = shp_counties.dissolve(by="state_abbr").reset_index()

    results = []
    for own_state in SAMPLE_STATES:
        nonlegal_states = states_poly[~states_poly.state_abbr.isin(legal_states)
                                       & (states_poly.state_abbr != own_state)]
        nonlegal_union = nonlegal_states.union_all()
        sub = counties[counties.state_abbr == own_state].copy()
        pts = gpd.GeoSeries([Point(xy) for xy in zip(sub["lon"], sub["lat"])], crs=4326).to_crs(5070)
        sub["dist_stateline_mi"] = pts.distance(nonlegal_union).to_numpy() / 1609.34
        results.append(sub)
    counties = pd.concat(results, ignore_index=True)

    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    nonlegal_zips_all = allzips[~allzips["state_id"].isin(legal_states)].copy()
    nz_lat, nz_lon, nz_pop = (nonlegal_zips_all["lat"].to_numpy(), nonlegal_zips_all["lng"].to_numpy(),
                               nonlegal_zips_all["population"].to_numpy())

    nonlegal_pop_within = np.zeros(len(counties))
    for i, (lat, lon) in enumerate(zip(counties["lat"].to_numpy(), counties["lon"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nonlegal_pop_within[i] = nz_pop[d <= threshold_mi].sum()
    counties[f"nonlegal_pop_within_{threshold_mi}mi"] = nonlegal_pop_within

    counties["border_county"] = ((counties["dist_stateline_mi"] <= threshold_mi) &
                                  (counties[f"nonlegal_pop_within_{threshold_mi}mi"] >= MIN_NONLEGAL_POP)).astype(int)

    out = os.path.join(TBL, f"county_border_stateline_{threshold_mi}mi.csv")
    counties[["fips5", "state_abbr", "county_name", "dist_stateline_mi",
              f"nonlegal_pop_within_{threshold_mi}mi", "border_county"]].to_csv(out, index=False)
    print(f"Wrote {out}")
    print(counties.groupby("state_abbr")["border_county"].agg(["sum", "size"]))


if __name__ == "__main__":
    threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    main(threshold)
