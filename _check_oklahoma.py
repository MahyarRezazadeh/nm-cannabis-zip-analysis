import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")

geo = pd.read_csv(os.path.join(TBL, "nm_dispensary_geocoded.csv"))
addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"), dtype={"zip5": str})
geo = geo.merge(addr[["dispensary_id", "zip5", "City"]], on="dispensary_id", how="left")
geo = geo.dropna(subset=["lat", "lon"]).copy()

SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
counties = counties.to_crs(5070)
states = counties.dissolve(by="state_abbr").reset_index()
ok = states[states.state_abbr == "OK"].geometry.iloc[0]
tx = states[states.state_abbr == "TX"].geometry.iloc[0]

pts = gpd.GeoSeries([Point(xy) for xy in zip(geo["lon"], geo["lat"])], crs=4326).to_crs(5070)
geo["dist_ok_mi"] = pts.distance(ok).to_numpy() / 1609.34
geo["dist_tx_mi"] = pts.distance(tx).to_numpy() / 1609.34
geo["ok_closer"] = geo["dist_ok_mi"] < geo["dist_tx_mi"]

print("Dispensaries where OK is closer than TX:", geo["ok_closer"].sum(), "of", len(geo))
print()
print("The 10 dispensaries closest to Oklahoma overall:")
print(geo.sort_values("dist_ok_mi")[["zip5", "City", "dist_ok_mi", "dist_tx_mi"]].head(10).to_string(index=False))
print()

# Union County NM is the one that actually touches OK's panhandle -- check its dispensaries specifically
union_zips = ["88414", "88415", "88401"]  # Clayton/Union county area zips seen earlier
print("Union County area (touches OK) dispensaries:")
print(geo[geo.zip5.isin(union_zips)][["zip5", "City", "dist_ok_mi", "dist_tx_mi"]].to_string(index=False))

# also print how much of the NM-OK vs NM-TX state line length differs, as context
nm = states[states.state_abbr == "NM"].geometry.iloc[0]
print()
print(f"NM-OK shared boundary length: {nm.boundary.intersection(ok.buffer(5000)).length/1609.34:.1f} mi (approx, 5km buffer)")
print(f"NM-TX shared boundary length: {nm.boundary.intersection(tx.buffer(5000)).length/1609.34:.1f} mi (approx, 5km buffer)")
