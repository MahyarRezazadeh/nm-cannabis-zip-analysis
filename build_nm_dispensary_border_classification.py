"""
Dispensary-level border classification for New Mexico, using the exact
same rule as the paper's main ZIP-code design (main.tex, equation
\\ref{baseline}): a location is BORDER if it sits within 75 miles of the
NM state line where NM borders a state with no legal recreational cannabis
market, provided at least 1,000 residents of that neighboring state live
within the same 75-mile radius.

The ZIP-level design applied this dispensary-by-dispensary and then took
"at least one qualifying dispensary in the ZIP" as the ZIP's border status
(see _check_option34_combined.py). At the dispensary level there's no
aggregation step needed -- each dispensary's own coordinates decide its
own border status directly, which is a strictly more precise version of
the same rule.

Coordinates come from tables/nm_dispensary_locations.csv (Census Batch
Geocoder + Nominatim fallback, 1,192 of 1,253 dispensaries / 95.1%
matched). Dispensaries without coordinates cannot be classified and are
dropped from the border design (flagged, not silently imputed).
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
EARTH_RADIUS_MI = 3958.8
THRESHOLD_MI = 75
MIN_NONLEGAL_POP = 1000


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def main():
    loc = pd.read_csv(os.path.join(TBL, "nm_dispensary_locations.csv"))
    geo = loc.dropna(subset=["lat", "lon"]).copy()
    n_dropped = len(loc) - len(geo)
    print(f"{len(geo)} of {len(loc)} dispensaries have coordinates and are classified "
          f"({n_dropped} dropped for missing coordinates)")

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    counties = counties.to_crs(5070)
    states = counties.dissolve(by="state_abbr").reset_index()
    nonlegal_states = states[~states.state_abbr.isin(legal_states) & (states.state_abbr != "NM")]
    nonlegal_union = nonlegal_states.union_all()
    print(f"Nonlegal neighbor states (as of {cutoff.date()}): "
          f"{sorted(set(states.state_abbr) - legal_states - {'NM'})}")

    pts = gpd.GeoSeries([Point(xy) for xy in zip(geo["lon"], geo["lat"])], crs=4326).to_crs(5070)
    geo["dist_stateline_mi"] = pts.distance(nonlegal_union).to_numpy() / 1609.34

    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    nonlegal_zips = allzips[~allzips["state_id"].isin(legal_states)]
    nz_lat, nz_lon, nz_pop = (nonlegal_zips["lat"].to_numpy(), nonlegal_zips["lng"].to_numpy(),
                               nonlegal_zips["population"].to_numpy())

    nonlegal_pop_within = np.zeros(len(geo))
    for i, (lat, lon) in enumerate(zip(geo["lat"].to_numpy(), geo["lon"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nonlegal_pop_within[i] = nz_pop[d <= THRESHOLD_MI].sum()
    geo["nonlegal_pop_within_75mi"] = nonlegal_pop_within

    geo["border_dispensary"] = ((geo["dist_stateline_mi"] <= THRESHOLD_MI) &
                                 (geo["nonlegal_pop_within_75mi"] >= MIN_NONLEGAL_POP)).astype(int)

    print(f"\n{geo['border_dispensary'].sum()} border dispensaries, "
          f"{(geo['border_dispensary'] == 0).sum()} interior dispensaries")

    out_cols = ["dispensary_id", "dist_stateline_mi", "nonlegal_pop_within_75mi", "border_dispensary"]
    out_path = os.path.join(TBL, "nm_dispensary_border_classification.csv")
    geo[out_cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
