"""
Build a quarterly ZIP-code panel for New Mexico dispensaries, per explicit
user request. Border definition: a ZIP code is "border" if it is within 50
miles of the nearest incorporated place (2023 population >= 5,000, same
threshold/source as the earlier 100-mile county-level check) located in a
state with no operating recreational retail market. This is a straight
distance-to-nonlegal-market rule -- ZIP codes near an already-LEGAL
neighbor (Arizona, Colorado) are NOT border, they fall into "interior"
along with everywhere else, exactly as instructed. Legal status is
evaluated as of NM's panel window (2022-2026); NM's actual bordering/nearby
states never change legal status within that window (OK and TX have never
legalized, UT has never legalized, AZ (2021) and CO (2014) were already
legal before NM's data begins), so this is a single, time-invariant
classification, not a per-year one.

ZIP-code centroid lat/lon and population come from a public ZCTA-level
reference table (simplemaps' free uszips.csv, ultimately sourced from
Census ACS 5-year estimates) since the Census Bureau's own ZCTA population
API requires a key not available in this session -- disclosed here, not
hidden. The Census Gazetteer ZCTA file (official, key-free) is used to
cross-check centroid coordinates agree.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
THRESHOLD_MI = 50
POP_MIN = 5000
EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def load_zip_reference():
    z = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    z = z.rename(columns={"zip": "zip5", "lat": "zip_lat", "lng": "zip_lon", "population": "zip_population"})
    return z[["zip5", "zip_lat", "zip_lon", "zip_population", "county_fips", "county_name", "state_id"]]


def load_nonlegal_cities():
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
    cutoff = pd.Timestamp("2023-12-31")  # any year within NM's constant-status panel window
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])

    nonlegal = cities[~cities["USPS"].isin(legal_states)].copy()
    nonlegal = nonlegal[nonlegal["POPESTIMATE2023"] >= POP_MIN].copy()
    print(f"Nonlegal-state city universe (pop>={POP_MIN}, as of 2023): {len(nonlegal)} cities, "
          f"{nonlegal['USPS'].nunique()} states: {sorted(nonlegal['USPS'].unique())}")
    return nonlegal


def main():
    zref = load_zip_reference()
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_panel_monthly_zip.csv"), dtype={"zip5": str})
    zref["zip5"] = zref["zip5"].astype(str).str.zfill(5)

    nm_zips = disp[["zip5"]].drop_duplicates().merge(zref, on="zip5", how="left")
    missing = nm_zips["zip_lat"].isna().sum()
    print(f"{len(nm_zips)} dispensary ZIP codes, {missing} unmatched to the ZIP reference table")
    if missing:
        print(nm_zips[nm_zips.zip_lat.isna()])

    cities = load_nonlegal_cities()
    dist_matrix = haversine_miles(
        nm_zips["zip_lat"].to_numpy()[:, None], nm_zips["zip_lon"].to_numpy()[:, None],
        cities["INTPTLAT"].to_numpy()[None, :], cities["INTPTLONG"].to_numpy()[None, :])
    nearest_idx = dist_matrix.argmin(axis=1)
    nm_zips["dist_nearest_nonlegal_city_mi"] = dist_matrix[np.arange(len(nm_zips)), nearest_idx]
    nm_zips["nearest_nonlegal_city"] = cities["NAME"].to_numpy()[nearest_idx]
    nm_zips["nearest_nonlegal_city_state"] = cities["USPS"].to_numpy()[nearest_idx]
    nm_zips["border_zip"] = (nm_zips["dist_nearest_nonlegal_city_mi"] <= THRESHOLD_MI).astype(int)

    print(f"\nBorder ZIP codes (<= {THRESHOLD_MI}mi): {nm_zips.border_zip.sum()} of {len(nm_zips)}")
    print(nm_zips.sort_values("dist_nearest_nonlegal_city_mi")[
        ["zip5", "county_name", "dist_nearest_nonlegal_city_mi", "nearest_nonlegal_city",
         "nearest_nonlegal_city_state", "border_zip"]].head(15).to_string(index=False))

    nm_zips.to_csv(os.path.join(TBL, "nm_zip_border_classification.csv"), index=False)

    # ---------- quarterly aggregation ----------
    disp["quarter"] = ((disp["month"] - 1) // 3) + 1
    q = disp.groupby(["zip5", "year", "quarter"]).agg(
        total_sales=("total_sales", "sum"),
        dispensary_count=("dispensary_id", "nunique"),
        n_months=("month", "nunique")).reset_index()
    q = q[q["n_months"] == 3].drop(columns="n_months")

    q = q.merge(nm_zips[["zip5", "zip_population", "county_fips", "county_name",
                          "dist_nearest_nonlegal_city_mi", "border_zip"]], on="zip5", how="left")
    q = q[q["zip_population"].notna() & (q["zip_population"] > 0)].copy()

    q["revenue_per_capita"] = q["total_sales"] / q["zip_population"]
    q["revenue_per_store"] = q["total_sales"] / q["dispensary_count"]
    q["dispensary_per_1000"] = 1000 * q["dispensary_count"] / q["zip_population"]
    for col in ["revenue_per_capita", "revenue_per_store", "dispensary_per_1000"]:
        q[f"log_{col}"] = np.log(q[col].where(q[col] > 0))
    q["log_population"] = np.log(q["zip_population"])
    q["state_abbr"] = "NM"
    q["yq"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)

    out = os.path.join(TBL, "nm_zip_quarterly_panel.csv")
    q.to_csv(out, index=False)
    print(f"\nWrote {out}: {q.shape}, {q.zip5.nunique()} ZIP codes, "
          f"{q.county_fips.nunique()} counties, periods {q.yq.min()}-{q.yq.max()}")
    print(f"Border ZIP-quarters: {q.border_zip.sum()} of {len(q)}")


if __name__ == "__main__":
    main()
