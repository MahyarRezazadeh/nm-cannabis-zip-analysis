"""
County-level border classification for CO, OR, MT, NM (WA already dropped
from this line of work), mirroring the methodology settled on for the NM
ZIP-code design: a county is BORDER if its centroid is within 75 miles of
the nonlegal-state line, provided at least 1,000 nonlegal-state residents
live within that same 75-mile reach (rules out empty/unpopulated stretches
of state line). County centroids substitute for geocoded dispensary
addresses here since address-level geocoding isn't uniformly available
across all 4 states (only CO and NM have reliable dispensary-level
geocoding in this project; OR's is flagged unreliable; MT has none) --
disclosed, not silently assumed away.

Legal-state status evaluated once as of 2023-12-31, the same convention
used for the NM ZIP design; time-varying entry-cap status is a SEPARATE
variable (is_capped, from the existing entry_capped column, state-year
level) not conflated with the geographic border classification.
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
SAMPLE_STATES = ["CO", "OR", "MT", "NM"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def main():
    cc = pd.read_stata(os.path.join(SOURCE_BASE, "data", "county_centroids.dta"))
    cc["fips5"] = cc["fips5"].astype(str).str.zfill(5)
    counties = cc[cc.state_abbr.isin(SAMPLE_STATES)].copy()
    print(f"{len(counties)} counties across {SAMPLE_STATES}")

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    # ---------- distance to nonlegal state line ----------
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

    # ---------- nonlegal-state population within 75mi (national ZIP reference) ----------
    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    nonlegal_zips_all = allzips[~allzips["state_id"].isin(legal_states)].copy()
    nz_lat, nz_lon, nz_pop = (nonlegal_zips_all["lat"].to_numpy(), nonlegal_zips_all["lng"].to_numpy(),
                               nonlegal_zips_all["population"].to_numpy())

    nonlegal_pop_within = np.zeros(len(counties))
    for i, (lat, lon) in enumerate(zip(counties["lat"].to_numpy(), counties["lon"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nonlegal_pop_within[i] = nz_pop[d <= THRESHOLD_MI].sum()
    counties["nonlegal_pop_within_75mi"] = nonlegal_pop_within

    counties["border_county"] = ((counties["dist_stateline_mi"] <= THRESHOLD_MI) &
                                  (counties["nonlegal_pop_within_75mi"] >= MIN_NONLEGAL_POP)).astype(int)

    out = os.path.join(TBL, "county_border_stateline.csv")
    counties[["fips5", "state_abbr", "county_name", "dist_stateline_mi", "nonlegal_pop_within_75mi",
              "border_county"]].to_csv(out, index=False)
    print(f"Wrote {out}")
    print(counties.groupby("state_abbr")["border_county"].agg(["sum", "size"]))


if __name__ == "__main__":
    main()
