"""
City-level border classification for OR and MT, mirroring the same rule
used throughout this project: a city is BORDER if its centroid (population
-weighted centroid of its own ZIP codes) is within 75 miles of the
nonlegal-state line, with >=1,000 nonlegal-state residents within that
same 75-mile reach. Legal-state status evaluated once as of 2023-12-31,
same convention as county_border_stateline.py.
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


def classify(state_abbr, panel_path, city_col="city"):
    panel = pd.read_csv(os.path.join(TBL, panel_path))
    cities = panel.drop_duplicates(city_col)[[city_col, "lat", "lon"]].reset_index(drop=True)

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    shp_counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    shp_counties = shp_counties.to_crs(5070)
    states_poly = shp_counties.dissolve(by="state_abbr").reset_index()
    nonlegal_states = states_poly[~states_poly.state_abbr.isin(legal_states) & (states_poly.state_abbr != state_abbr)]
    nonlegal_union = nonlegal_states.union_all()
    print(f"{state_abbr}: nonlegal neighbor pool includes "
          f"{sorted(set(states_poly.state_abbr) - legal_states - {state_abbr})}")

    pts = gpd.GeoSeries([Point(xy) for xy in zip(cities["lon"], cities["lat"])], crs=4326).to_crs(5070)
    cities["dist_stateline_mi"] = pts.distance(nonlegal_union).to_numpy() / 1609.34

    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    nonlegal_zips = allzips[~allzips["state_id"].isin(legal_states)]
    nz_lat, nz_lon, nz_pop = (nonlegal_zips["lat"].to_numpy(), nonlegal_zips["lng"].to_numpy(),
                               nonlegal_zips["population"].to_numpy())

    nonlegal_pop_within = np.zeros(len(cities))
    for i, (lat, lon) in enumerate(zip(cities["lat"].to_numpy(), cities["lon"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nonlegal_pop_within[i] = nz_pop[d <= THRESHOLD_MI].sum()
    cities["nonlegal_pop_within_75mi"] = nonlegal_pop_within

    cities["border_city"] = ((cities["dist_stateline_mi"] <= THRESHOLD_MI) &
                              (cities["nonlegal_pop_within_75mi"] >= MIN_NONLEGAL_POP)).astype(int)

    out = os.path.join(TBL, f"{state_abbr.lower()}_city_border_classification.csv")
    cities.to_csv(out, index=False)
    print(f"{state_abbr}: {cities.border_city.sum()} of {len(cities)} cities classified border")
    print(f"Wrote {out}\n")
    return cities


if __name__ == "__main__":
    classify("OR", "or_city_quarterly_panel.csv")
    classify("MT", "mt_city_quarterly_panel.csv")
