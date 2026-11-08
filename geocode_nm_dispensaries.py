"""
Geocode NM dispensary addresses via the Census Bureau's free Batch Geocoder
(https://geocoding.geo.census.gov/geocoder/), the same free/keyless service
used earlier this project for CO/OR (dispensaries_master_geocoded.csv,
geocode_source="census_street"). No API key required. Batch endpoint takes
a headerless CSV (id, street, city, state, zip) as multipart form data and
returns match status + lat/lon for each input row.
"""
import os
import time
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
BENCHMARK = "Public_AR_Current"


def main():
    addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"), dtype={"zip5": str})
    input_df = addr[["dispensary_id", "addr_norm", "City", "State", "zip5"]].copy()
    input_csv_path = os.path.join(TBL, "nm_geocode_input.csv")
    input_df.to_csv(input_csv_path, index=False, header=False)
    print(f"Wrote {input_csv_path}: {len(input_df)} rows to geocode")

    all_results = []
    chunk_size = 1000  # Census batch geocoder practical limit per request is 10000, but keep requests smaller/safer
    for start in range(0, len(input_df), chunk_size):
        chunk = input_df.iloc[start:start + chunk_size]
        chunk_path = os.path.join(TBL, f"_geocode_chunk_{start}.csv")
        chunk.to_csv(chunk_path, index=False, header=False)

        print(f"Submitting rows {start}-{start + len(chunk)}...")
        with open(chunk_path, "rb") as f:
            files = {"addressFile": (os.path.basename(chunk_path), f, "text/csv")}
            data = {"benchmark": BENCHMARK}
            for attempt in range(3):
                try:
                    r = requests.post(URL, files=files, data=data, timeout=180)
                    r.raise_for_status()
                    break
                except Exception as e:
                    print(f"  attempt {attempt + 1} failed: {e}")
                    time.sleep(5)
            else:
                raise RuntimeError(f"Failed to geocode chunk starting at {start}")

        result_path = os.path.join(TBL, f"_geocode_result_{start}.csv")
        with open(result_path, "wb") as f:
            f.write(r.content)

        cols = ["dispensary_id", "input_address", "match_status", "match_type",
                "matched_address", "lonlat", "tiger_line_id", "side"]
        res = pd.read_csv(result_path, header=None, names=cols, dtype={"dispensary_id": str})
        all_results.append(res)
        os.remove(chunk_path)
        print(f"  {res['match_status'].value_counts().to_dict()}")

    result = pd.concat(all_results, ignore_index=True)
    result["dispensary_id"] = result["dispensary_id"].astype(int)
    split = result["lonlat"].str.split(",", expand=True)
    result["lon"] = pd.to_numeric(split[0], errors="coerce")
    result["lat"] = pd.to_numeric(split[1], errors="coerce")

    out_path = os.path.join(TBL, "nm_dispensary_geocoded.csv")
    result.to_csv(out_path, index=False)
    matched = result["lat"].notna().sum()
    print(f"\nWrote {out_path}: {len(result)} rows, {matched} successfully geocoded "
          f"({100 * matched / len(result):.1f}%)")


if __name__ == "__main__":
    main()
