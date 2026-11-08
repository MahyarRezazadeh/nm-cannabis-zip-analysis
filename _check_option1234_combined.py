import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def cities_universe(pop_min, legal_states):
    gaz = pd.read_csv(os.path.join(SOURCE_BASE, "data", "2023_Gaz_place_national.txt"), sep="\t")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["state_fips"] = (gaz["GEOID"] // 100000).astype(int)
    gaz["place_fips"] = (gaz["GEOID"] % 100000).astype(int)
    pop = pd.read_csv(os.path.join(SOURCE_BASE, "data", "sub-est2023.csv"), encoding="latin1")
    pop = pop[pop["SUMLEV"] == 162][["STATE", "PLACE", "POPESTIMATE2023"]].copy()
    pop = pop.rename(columns={"STATE": "state_fips", "PLACE": "place_fips"})
    cities = gaz.merge(pop, on=["state_fips", "place_fips"], how="inner")
    return cities[~cities["USPS"].isin(legal_states) & (cities["POPESTIMATE2023"] >= pop_min)].copy()


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

    geo = pd.read_csv(os.path.join(TBL, "nm_dispensary_geocoded.csv"))
    addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"), dtype={"zip5": str})
    geo = geo.merge(addr[["dispensary_id", "zip5"]], on="dispensary_id", how="left")
    geo = geo.dropna(subset=["lat", "lon"]).copy()

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    # ---------- option 3+4 term (already built) ----------
    prev = pd.read_csv(os.path.join(TBL, "nm_zip_border_option34.csv"), dtype={"zip5": str})
    # rebuild the per-dispensary qualifies flag for option 3+4 fresh (same logic as before)
    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    counties = counties.to_crs(5070)
    states = counties.dissolve(by="state_abbr").reset_index()
    nonlegal_states = states[~states.state_abbr.isin(legal_states) & (states.state_abbr != "NM")]
    nonlegal_union = nonlegal_states.union_all()
    pts = gpd.GeoSeries([Point(xy) for xy in zip(geo["lon"], geo["lat"])], crs=4326).to_crs(5070)
    geo["dist_stateline_mi"] = pts.distance(nonlegal_union).to_numpy() / 1609.34

    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    nonlegal_zips = allzips[~allzips["state_id"].isin(legal_states)].copy()
    nz_lat, nz_lon, nz_pop = nonlegal_zips["lat"].to_numpy(), nonlegal_zips["lng"].to_numpy(), nonlegal_zips["population"].to_numpy()
    nonlegal_pop_within = np.zeros(len(geo))
    for i, (lat, lon) in enumerate(zip(geo["lat"].to_numpy(), geo["lon"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nonlegal_pop_within[i] = nz_pop[d <= 75].sum()
    geo["nonlegal_pop_within_75mi"] = nonlegal_pop_within
    geo["qualifies_34"] = ((geo["dist_stateline_mi"] <= 75) & (geo["nonlegal_pop_within_75mi"] >= 1000)).astype(int)

    # ---------- option 1: city (pop>=5000) within 75mi ----------
    cities5000 = cities_universe(5000, legal_states)
    dm1 = haversine_miles(geo["lat"].to_numpy()[:, None], geo["lon"].to_numpy()[:, None],
                           cities5000["INTPTLAT"].to_numpy()[None, :], cities5000["INTPTLONG"].to_numpy()[None, :])
    geo["dist_city5000_mi"] = dm1.min(axis=1)
    geo["qualifies_1"] = (geo["dist_city5000_mi"] <= 75).astype(int)

    # ---------- option 2: city (pop>=1000) within 50mi ----------
    cities1000 = cities_universe(1000, legal_states)
    dm2 = haversine_miles(geo["lat"].to_numpy()[:, None], geo["lon"].to_numpy()[:, None],
                           cities1000["INTPTLAT"].to_numpy()[None, :], cities1000["INTPTLONG"].to_numpy()[None, :])
    geo["dist_city1000_mi"] = dm2.min(axis=1)
    geo["qualifies_2"] = (geo["dist_city1000_mi"] <= 50).astype(int)

    # ---------- combined: any of 1, 2, or 3+4 qualifies ----------
    geo["qualifies_any"] = geo[["qualifies_34", "qualifies_1", "qualifies_2"]].max(axis=1)

    g = geo.groupby("zip5").agg(
        n=("qualifies_any", "size"),
        n_q34=("qualifies_34", "sum"), n_q1=("qualifies_1", "sum"), n_q2=("qualifies_2", "sum"),
        n_qany=("qualifies_any", "sum")).reset_index()
    g["border_zip"] = (g["n_qany"] >= 1).astype(int)

    new_border = set(g[g.border_zip == 1]["zip5"]) & reg_zips
    prev_border = set(prev[prev.border_zip == 1]["zip5"]) & reg_zips
    added_vs_prev = new_border - prev_border
    added_vs_baseline = new_border - baseline_border

    print(f"BASELINE (geocoded, majority, city-50mi-pop5000): {len(baseline_border)} border ZIPs")
    print(f"PRIOR (option 3+4 only): {len(prev_border)} border ZIPs")
    print(f"NOW (option 1+2+3+4 combined): {len(new_border)} border ZIPs in regression sample")
    print(f"  Added vs. option-3+4-only: +{len(added_vs_prev)} -- {sorted(added_vs_prev)}")
    print(f"  Added vs. original baseline: +{len(added_vs_baseline)}")

    g.to_csv(os.path.join(TBL, "nm_zip_border_option1234.csv"), index=False)


if __name__ == "__main__":
    main()
