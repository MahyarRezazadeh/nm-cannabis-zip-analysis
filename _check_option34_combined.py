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


def regression_sample_zips():
    panel = pd.read_csv(os.path.join(TBL, "nm_zip_quarterly_panel.csv"), dtype={"zip5": str, "county_fips": str})
    panel["county_fips_clean"] = panel["county_fips"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    countypop = pd.read_csv(os.path.join(SOURCE_BASE, "tables", "master_percapita_panel.csv"), dtype={"fips5": str})
    nm_fips = set(countypop[countypop.state_abbr == "NM"]["fips5"])
    return set(panel[panel.county_fips_clean.isin(nm_fips)]["zip5"].unique())


def main():
    baseline = pd.read_csv(os.path.join(TBL, "nm_zip_border_refined.csv"), dtype={"zip5": str})
    reg_zips = regression_sample_zips()
    baseline_border = set(baseline[(baseline.border_zip_refined == 1) & (baseline.zip5.isin(reg_zips))]["zip5"])
    print(f"BASELINE: {len(baseline_border)} border ZIPs")

    geo = pd.read_csv(os.path.join(TBL, "nm_dispensary_geocoded.csv"))
    addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"), dtype={"zip5": str})
    geo = geo.merge(addr[["dispensary_id", "zip5"]], on="dispensary_id", how="left")
    geo = geo.dropna(subset=["lat", "lon"]).copy()

    # ---------- legal-state set (same convention as elsewhere) ----------
    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    # ---------- distance to nonlegal state line (option 4) ----------
    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    counties = counties.to_crs(5070)
    states = counties.dissolve(by="state_abbr").reset_index()
    nonlegal_states = states[~states.state_abbr.isin(legal_states) & (states.state_abbr != "NM")]
    nonlegal_union = nonlegal_states.union_all()

    pts = gpd.GeoSeries([Point(xy) for xy in zip(geo["lon"], geo["lat"])], crs=4326).to_crs(5070)
    geo["dist_stateline_mi"] = pts.distance(nonlegal_union).to_numpy() / 1609.34

    # ---------- nonlegal-state population within 75mi (national ZIP reference) ----------
    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    nonlegal_zips = allzips[allzips["state_id"].isin(legal_states) == False].copy()
    print(f"Nonlegal-state ZIP universe: {len(nonlegal_zips)} of {len(allzips)} national ZIPs")

    nz_lat = nonlegal_zips["lat"].to_numpy()
    nz_lon = nonlegal_zips["lng"].to_numpy()
    nz_pop = nonlegal_zips["population"].to_numpy()

    nonlegal_pop_within = np.zeros(len(geo))
    for i, (lat, lon) in enumerate(zip(geo["lat"].to_numpy(), geo["lon"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nonlegal_pop_within[i] = nz_pop[d <= THRESHOLD_MI].sum()
    geo["nonlegal_pop_within_75mi"] = nonlegal_pop_within

    # ---------- combined rule: within 75mi of state line AND >=1000 nonlegal pop within 75mi ----------
    geo["qualifies"] = ((geo["dist_stateline_mi"] <= THRESHOLD_MI) &
                         (geo["nonlegal_pop_within_75mi"] >= MIN_NONLEGAL_POP)).astype(int)

    # option 3 rule: ANY qualifying dispensary makes the ZIP border
    g = geo.groupby("zip5").agg(n=("qualifies", "size"), n_qualify=("qualifies", "sum")).reset_index()
    g["border_zip"] = (g["n_qualify"] >= 1).astype(int)

    new_border = set(g[g.border_zip == 1]["zip5"]) & reg_zips
    added = new_border - baseline_border
    removed = baseline_border - new_border
    print(f"\nCOMBINED (any-dispensary rule + state-line distance <=75mi + nonlegal pop >= {MIN_NONLEGAL_POP} "
          f"within 75mi):")
    print(f"  {len(new_border)} border ZIPs in regression sample (+{len(added)}, -{len(removed)} vs baseline)")
    print(f"  Added: {sorted(added)}")
    print(f"  Removed: {sorted(removed)}")

    # show the excluded-for-low-population cases for transparency
    excluded_lowpop = geo[(geo.dist_stateline_mi <= THRESHOLD_MI) & (geo.nonlegal_pop_within_75mi < MIN_NONLEGAL_POP)]
    print(f"\nDispensaries within {THRESHOLD_MI}mi of the state line but excluded for low nonlegal population "
          f"(<{MIN_NONLEGAL_POP}): {len(excluded_lowpop)}")
    if len(excluded_lowpop):
        print(excluded_lowpop[["zip5", "dist_stateline_mi", "nonlegal_pop_within_75mi"]].drop_duplicates("zip5")
              .to_string(index=False))

    g.to_csv(os.path.join(TBL, "nm_zip_border_option34.csv"), index=False)


if __name__ == "__main__":
    main()
