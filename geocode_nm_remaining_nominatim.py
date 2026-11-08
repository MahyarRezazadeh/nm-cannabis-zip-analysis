"""
Second-pass geocoding for the 173 NM dispensaries the Census Batch Geocoder
could not resolve (147 No_Match + 26 Tie), using OpenStreetMap's Nominatim
search API as a fallback. Free, no API key, but rate-limited to 1 req/sec
per Nominatim's usage policy (https://operations.osmfoundation.org/policies/nominatim/)
-- a valid identifying User-Agent is required or requests get blocked.

Falls back from a full street+city+state+zip query to a looser
street+state query if the first attempt returns nothing, since a few
addresses have typos/formatting Nominatim's parser trips on with the zip
appended.
"""
import os
import time
import requests
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
HEADERS = {"User-Agent": "nm-dispensary-research/1.0 (academic research project)"}
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def query(q):
    try:
        r = requests.get(NOMINATIM_URL, params={"q": q, "format": "json", "limit": 1,
                                                  "countrycodes": "us"}, headers=HEADERS, timeout=10)
        r.raise_for_status()
        results = r.json()
        if results:
            return float(results[0]["lon"]), float(results[0]["lat"]), results[0].get("display_name", "")
    except requests.RequestException as e:
        print(f"  request error: {e}")
    return None, None, None


def main():
    loc = pd.read_csv(os.path.join(TBL, "nm_dispensary_locations.csv"))
    old = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"))
    missing = loc[loc.lat.isna()][["dispensary_id", "addr_norm"]].merge(
        old.drop(columns=["dispensary_id"]), on="addr_norm", how="left")

    print(f"{len(missing)} addresses to geocode via Nominatim")
    results = []
    for i, row in enumerate(missing.itertuples(), 1):
        q_full = f"{row.addr_norm}, {row.City}, {row.State} {row.zip5}"
        lon, lat, matched = query(q_full)
        time.sleep(1.1)
        attempt = "street+city+state+zip"
        if lon is None:
            q_loose = f"{row.addr_norm}, {row.State}"
            lon, lat, matched = query(q_loose)
            time.sleep(1.1)
            attempt = "street+state"
            if lon is None:
                attempt = "no_match"
        results.append({"dispensary_id": row.dispensary_id, "addr_norm": row.addr_norm,
                         "nominatim_lon": lon, "nominatim_lat": lat,
                         "nominatim_matched_display": matched, "nominatim_attempt": attempt})
        if i % 20 == 0 or i == len(missing):
            found = sum(1 for r in results if r["nominatim_lon"] is not None)
            print(f"  {i}/{len(missing)} processed, {found} matched so far")

    out = pd.DataFrame(results)
    out_path = os.path.join(TBL, "nm_dispensary_nominatim_fallback.csv")
    out.to_csv(out_path, index=False)
    n_matched = out["nominatim_lon"].notna().sum()
    print(f"\nNominatim matched {n_matched} of {len(out)} remaining addresses")
    print(out["nominatim_attempt"].value_counts())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
