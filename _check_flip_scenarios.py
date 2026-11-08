import os
import numpy as np
import pandas as pd

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


def load_geocoded():
    geo = pd.read_csv(os.path.join(TBL, "nm_dispensary_geocoded.csv"))
    addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"), dtype={"zip5": str})
    geo = geo.merge(addr[["dispensary_id", "zip5"]], on="dispensary_id", how="left")
    return geo.dropna(subset=["lat", "lon"]).copy()


def regression_sample_zips():
    """ZIP codes that actually match an NM county population (the real regression sample)."""
    panel = pd.read_csv(os.path.join(TBL, "nm_zip_quarterly_panel.csv"), dtype={"zip5": str, "county_fips": str})
    panel["county_fips_clean"] = panel["county_fips"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    countypop = pd.read_csv(os.path.join(SOURCE_BASE, "tables", "master_percapita_panel.csv"), dtype={"fips5": str})
    nm_fips = set(countypop[countypop.state_abbr == "NM"]["fips5"])
    matched = panel[panel.county_fips_clean.isin(nm_fips)]["zip5"].unique()
    return set(matched)


def cities_universe(pop_min):
    gaz = pd.read_csv(os.path.join(SOURCE_BASE, "data", "2023_Gaz_place_national.txt"), sep="\t")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["state_fips"] = (gaz["GEOID"] // 100000).astype(int)
    gaz["place_fips"] = (gaz["GEOID"] % 100000).astype(int)
    pop = pd.read_csv(os.path.join(SOURCE_BASE, "data", "sub-est2023.csv"), encoding="latin1")
    pop = pop[pop["SUMLEV"] == 162][["STATE", "PLACE", "POPESTIMATE2023"]].copy()
    pop = pop.rename(columns={"STATE": "state_fips", "PLACE": "place_fips"})
    cities = gaz.merge(pop, on=["state_fips", "place_fips"], how="inner")

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])
    nonlegal = cities[~cities["USPS"].isin(legal_states) & (cities["POPESTIMATE2023"] >= pop_min)].copy()
    return nonlegal


def dist_to_nearest_city(geo, cities):
    dm = haversine_miles(geo["lat"].to_numpy()[:, None], geo["lon"].to_numpy()[:, None],
                          cities["INTPTLAT"].to_numpy()[None, :], cities["INTPTLONG"].to_numpy()[None, :])
    return dm.min(axis=1)


def zip_majority_border(geo, dist_col, threshold, rule="majority"):
    geo = geo.copy()
    geo["is_border"] = (geo[dist_col] <= threshold).astype(int)
    g = geo.groupby("zip5").agg(n=("is_border", "size"), n_border=("is_border", "sum")).reset_index()
    if rule == "majority":
        g["border_zip"] = (g["n_border"] >= g["n"] / 2).astype(int)
    else:  # "any"
        g["border_zip"] = (g["n_border"] >= 1).astype(int)
    return g


def main():
    baseline = pd.read_csv(os.path.join(TBL, "nm_zip_border_refined.csv"), dtype={"zip5": str})
    reg_zips = regression_sample_zips()
    baseline_border = set(baseline[(baseline.border_zip_refined == 1) & (baseline.zip5.isin(reg_zips))]["zip5"])
    print(f"BASELINE: {len(baseline_border)} border ZIP codes in the regression sample "
          f"({len(reg_zips)} total matched ZIPs)")
    print()

    geo = load_geocoded()
    cities50_5000 = cities_universe(5000)

    # ---------- Option 1: raise distance threshold ----------
    geo["dist_5000pop"] = dist_to_nearest_city(geo, cities50_5000)
    for thresh in [75, 100]:
        g = zip_majority_border(geo, "dist_5000pop", thresh, "majority")
        new_border = set(g[g.border_zip == 1]["zip5"]) & reg_zips
        added = new_border - baseline_border
        print(f"OPTION 1 ({thresh}mi threshold, majority vote): {len(new_border)} border ZIPs "
              f"(+{len(added)} vs baseline) -- added: {sorted(added)}")
    print()

    # ---------- Option 2: lower population threshold ----------
    for pop_min in [2500, 1000]:
        cities_lowpop = cities_universe(pop_min)
        geo[f"dist_{pop_min}pop"] = dist_to_nearest_city(geo, cities_lowpop)
        g = zip_majority_border(geo, f"dist_{pop_min}pop", 50, "majority")
        new_border = set(g[g.border_zip == 1]["zip5"]) & reg_zips
        added = new_border - baseline_border
        print(f"OPTION 2 (pop>={pop_min} city threshold, 50mi, majority): {len(new_border)} border ZIPs "
              f"(+{len(added)} vs baseline) -- added: {sorted(added)}")
    print()

    # ---------- Option 3: "any dispensary" instead of majority vote ----------
    g = zip_majority_border(geo, "dist_5000pop", 50, "any")
    new_border = set(g[g.border_zip == 1]["zip5"]) & reg_zips
    added = new_border - baseline_border
    print(f"OPTION 3 (any-dispensary rule, 50mi, pop>=5000): {len(new_border)} border ZIPs "
          f"(+{len(added)} vs baseline) -- added: {sorted(added)}")
    print()

    # ---------- Option 4: distance to nonlegal state line (not city) ----------
    import geopandas as gpd
    from shapely.geometry import Point

    SHP = os.path.join(SOURCE_BASE, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")
    counties = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"})
    counties = counties.to_crs(5070)
    states = counties.dissolve(by="state_abbr").reset_index()

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])
    nonlegal_states = states[~states.state_abbr.isin(legal_states) & (states.state_abbr != "NM")]
    nonlegal_union = nonlegal_states.union_all()

    pts = gpd.GeoSeries([Point(xy) for xy in zip(geo["lon"], geo["lat"])], crs=4326).to_crs(5070)
    dist_m = pts.distance(nonlegal_union)
    geo["dist_stateline_mi"] = dist_m.to_numpy() / 1609.34

    for thresh in [50, 75, 100]:
        g = zip_majority_border(geo, "dist_stateline_mi", thresh, "majority")
        new_border = set(g[g.border_zip == 1]["zip5"]) & reg_zips
        added = new_border - baseline_border
        print(f"OPTION 4 ({thresh}mi to nonlegal STATE LINE, majority): {len(new_border)} border ZIPs "
              f"(+{len(added)} vs baseline) -- added: {sorted(added)}")


if __name__ == "__main__":
    main()
