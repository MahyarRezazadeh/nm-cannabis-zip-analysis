"""
Refine the ZIP-level border classification using actual geocoded dispensary
locations instead of the ZIP centroid, per explicit user request for a
finer geocode level. For each of the 1,082 successfully geocoded
dispensaries (86.2% match rate via the Census Batch Geocoder), compute
distance to the nearest nonlegal-state city (same 50mi / population>=5,000
rule as the ZIP-centroid design). A ZIP code's refined border status is the
MAJORITY vote across its own geocoded dispensaries (ties broken toward
border, the more conservative/inclusive direction); ZIP codes with zero
successfully-geocoded dispensaries keep their original centroid-based
classification (disclosed, not silently dropped).
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
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff), "state_abbr"])
    nonlegal = cities[~cities["USPS"].isin(legal_states) & (cities["POPESTIMATE2023"] >= POP_MIN)].copy()
    return nonlegal


def main():
    geo = pd.read_csv(os.path.join(TBL, "nm_dispensary_geocoded.csv"))
    addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"), dtype={"zip5": str})
    geo = geo.merge(addr[["dispensary_id", "zip5"]], on="dispensary_id", how="left")
    geo = geo.dropna(subset=["lat", "lon"]).copy()
    print(f"{len(geo)} of {len(addr)} dispensaries successfully geocoded "
          f"({100 * len(geo) / len(addr):.1f}%)")

    cities = load_nonlegal_cities()
    dm = haversine_miles(geo["lat"].to_numpy()[:, None], geo["lon"].to_numpy()[:, None],
                          cities["INTPTLAT"].to_numpy()[None, :], cities["INTPTLONG"].to_numpy()[None, :])
    nearest_idx = dm.argmin(axis=1)
    geo["dist_nearest_nonlegal_city_mi"] = dm[np.arange(len(geo)), nearest_idx]
    geo["nearest_nonlegal_city"] = cities["NAME"].to_numpy()[nearest_idx]
    geo["nearest_nonlegal_city_state"] = cities["USPS"].to_numpy()[nearest_idx]
    geo["border_dispensary"] = (geo["dist_nearest_nonlegal_city_mi"] <= THRESHOLD_MI).astype(int)

    geo.to_csv(os.path.join(TBL, "nm_dispensary_geocoded_border.csv"), index=False)
    print(f"Border dispensaries (geocoded, <= {THRESHOLD_MI}mi): {geo.border_dispensary.sum()} of {len(geo)}")

    # ---------- majority vote up to ZIP ----------
    zip_vote = geo.groupby("zip5").agg(
        n_geocoded=("dispensary_id", "size"), n_border=("border_dispensary", "sum")).reset_index()
    zip_vote["border_zip_geocoded"] = (zip_vote["n_border"] >= zip_vote["n_geocoded"] / 2).astype(int)

    orig = pd.read_csv(os.path.join(TBL, "nm_zip_border_classification.csv"), dtype={"zip5": str})
    orig = orig[["zip5", "border_zip"]].rename(columns={"border_zip": "border_zip_centroid"})
    merged = orig.merge(zip_vote, on="zip5", how="left")
    merged["border_zip_refined"] = merged["border_zip_geocoded"]
    fallback = merged["border_zip_refined"].isna()
    merged.loc[fallback, "border_zip_refined"] = merged.loc[fallback, "border_zip_centroid"]
    merged["border_zip_refined"] = merged["border_zip_refined"].astype(int)
    print(f"\n{fallback.sum()} of {len(merged)} ZIP codes had no successfully-geocoded dispensary "
          f"-- kept centroid-based classification")

    changed = merged[merged["border_zip_centroid"] != merged["border_zip_refined"]]
    print(f"\n{len(changed)} ZIP codes changed classification vs. the centroid-based design:")
    print(changed[["zip5", "border_zip_centroid", "border_zip_refined", "n_geocoded", "n_border"]]
          .to_string(index=False))

    out = os.path.join(TBL, "nm_zip_border_refined.csv")
    merged.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"Refined classification: {merged.border_zip_refined.sum()} border, "
          f"{(merged.border_zip_refined == 0).sum()} interior (of {len(merged)})")


if __name__ == "__main__":
    main()
