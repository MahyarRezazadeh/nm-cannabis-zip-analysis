import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")

geo = pd.read_csv(os.path.join(TBL, "nm_dispensary_geocoded.csv"))
addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"), dtype={"zip5": str})
geo = geo.merge(addr[["dispensary_id", "zip5"]], on="dispensary_id", how="left")
geo = geo.dropna(subset=["lat", "lon"]).copy()

border34 = pd.read_csv(os.path.join(TBL, "nm_zip_border_option34.csv"), dtype={"zip5": str})
border_zips = set(border34[border34.border_zip == 1]["zip5"])
geo = geo[geo.zip5.isin(border_zips)].copy()

legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
cutoff = pd.Timestamp("2023-12-31")
legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
counties = counties.to_crs(5070)
states = counties.dissolve(by="state_abbr").reset_index()
nonlegal_states = states[~states.state_abbr.isin(legal_states) & (states.state_abbr != "NM")].reset_index(drop=True)

pts = gpd.GeoSeries([Point(xy) for xy in zip(geo["lon"], geo["lat"])], crs=4326).to_crs(5070)

# distance to each nonlegal state individually; keep the nearest one's name
import numpy as np
dists = np.zeros((len(pts), len(nonlegal_states)))
for j, row in nonlegal_states.iterrows():
    dists[:, j] = pts.distance(row.geometry).to_numpy() / 1609.34

nearest_idx = dists.argmin(axis=1)
geo["nearest_nonlegal_state"] = nonlegal_states["state_abbr"].to_numpy()[nearest_idx]
geo["dist_to_nearest_state_mi"] = dists[np.arange(len(geo)), nearest_idx]

# restrict to only the QUALIFYING dispensary-state pair (<=75mi to that state's own line),
# not every dispensary that merely happens to sit in an already-qualifying ZIP
geo = geo[geo["dist_to_nearest_state_mi"] <= 75].copy()
print(f"Restricting to dispensaries actually within 75mi of their nearest nonlegal state: {len(geo)} rows\n")

print("Nearest nonlegal state, by dispensary (border ZIPs only):")
print(geo.groupby("nearest_nonlegal_state").agg(
    n_dispensaries=("dispensary_id", "size"), n_zips=("zip5", "nunique"),
    mean_dist=("dist_to_nearest_state_mi", "mean")).sort_values("n_dispensaries", ascending=False))

print()
print("Which ZIP codes pair with which nonlegal state:")
print(geo.groupby(["nearest_nonlegal_state", "zip5"]).size().reset_index(name="n").to_string(index=False))
