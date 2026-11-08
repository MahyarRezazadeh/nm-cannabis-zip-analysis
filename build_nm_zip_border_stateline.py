"""
ZIP-level border classification for New Mexico, using the SAME methodology
as this project's county-level border_stateline files: each ZIP's own
centroid distance (via a projected CRS) to the dissolved polygon of
nonlegal neighboring states, 75-mile threshold, plus a >=1,000
nonlegal-state population requirement within that same 75-mile reach
(ruling out empty desert stretches). This is a fresh, geography-based
classification for the FULL universe of populated NM ZIP codes (not just
the ~120 ZIPs that happen to contain a dispensary), needed for the
non-cannabis-sector placebo check at ZIP level.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
THRESHOLD_MI = 75
MIN_NONLEGAL_POP = 1000
EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def main():
    zips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"),
                        dtype={"zip": str, "county_fips": str})
    nm = zips[zips.state_id == "NM"].dropna(subset=["lat", "lng", "population"]).copy()
    nm = nm[nm.population > 0]
    print(f"{len(nm)} populated NM ZIPs")

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    shp_counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    shp_counties = shp_counties.to_crs(5070)
    states_poly = shp_counties.dissolve(by="state_abbr").reset_index()

    nonlegal_states = states_poly[~states_poly.state_abbr.isin(legal_states) & (states_poly.state_abbr != "NM")]
    nonlegal_union = nonlegal_states.union_all()

    pts = gpd.GeoSeries([Point(xy) for xy in zip(nm["lng"], nm["lat"])], crs=4326).to_crs(5070)
    nm["dist_stateline_mi"] = pts.distance(nonlegal_union).to_numpy() / 1609.34

    allzips = zips.dropna(subset=["lat", "lng", "population"])
    nonlegal_zips_all = allzips[~allzips["state_id"].isin(legal_states)].copy()
    nz_lat, nz_lon, nz_pop = (nonlegal_zips_all["lat"].to_numpy(), nonlegal_zips_all["lng"].to_numpy(),
                               nonlegal_zips_all["population"].to_numpy())

    nonlegal_pop_within = np.zeros(len(nm))
    for i, (lat, lon) in enumerate(zip(nm["lat"].to_numpy(), nm["lng"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nonlegal_pop_within[i] = nz_pop[d <= THRESHOLD_MI].sum()
    nm["nonlegal_pop_within_75mi"] = nonlegal_pop_within

    nm["border_zip"] = ((nm["dist_stateline_mi"] <= THRESHOLD_MI) &
                         (nm["nonlegal_pop_within_75mi"] >= MIN_NONLEGAL_POP)).astype(int)

    out = nm.rename(columns={"zip": "zip5", "lat": "zip_lat", "lng": "zip_lon", "population": "zip_population"})
    out = out[["zip5", "zip_lat", "zip_lon", "zip_population", "county_fips", "county_name",
               "dist_stateline_mi", "nonlegal_pop_within_75mi", "border_zip"]]
    out_path = os.path.join(TBL, "nm_zip_border_stateline.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out["border_zip"].value_counts())


if __name__ == "__main__":
    main()
