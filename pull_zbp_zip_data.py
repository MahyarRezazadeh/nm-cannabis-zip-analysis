"""
Pulls ZIP Code Business Patterns (ZBP, via the Census Bureau's CBP API)
establishment counts for the three non-cannabis placebo sectors, all
years 2018-2023 (2024 not yet released as of this pull), nationally in
one call per (year, NAICS code) -- the ZIP-level CBP endpoint does not
support a state-scoped wildcard query ("unknown/unsupported geography
hierarchy"), so we pull national and filter to New Mexico ZIPs locally.

This gives an establishment-COUNT outcome (not gross receipts, which
ZBP does not report) -- i.e., a density-style placebo, directly
analogous to the "dispensary density" outcome that carries this paper's
strongest and most significant cannabis border result.
"""
import os
import time
import json
import urllib.request
import urllib.error
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
KEY_PATH = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\marijuana_new\Economic Letters\data\census_api_key.txt"

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
NAICS_CODES = {"445310": "alcohol", "4451": "grocery", "44-45": "total_retail"}

with open(KEY_PATH) as f:
    API_KEY = f.read().strip()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def pull_year_naics(year, naics_code):
    url = (f"https://api.census.gov/data/{year}/cbp?get=ESTAB"
           f"&for=zip%20code:*&NAICS2017={naics_code}&key={API_KEY}")
    for attempt in range(4):
        status, body = get(url)
        if status == 200:
            data = json.loads(body)
            df = pd.DataFrame(data[1:], columns=data[0])
            df["ESTAB"] = pd.to_numeric(df["ESTAB"], errors="coerce")
            return df
        print(f"  status={status} year={year} naics={naics_code}, attempt {attempt+1}, retrying...")
        time.sleep(10)
    print(f"  GAVE UP year={year} naics={naics_code}")
    return None


def main():
    nm_zips = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    nm_zip_set = set(nm_zips["zip5"])

    all_rows = []
    for year in YEARS:
        for code, sector in NAICS_CODES.items():
            print(f"Pulling {year} {sector} ({code})...")
            df = pull_year_naics(year, code)
            if df is None:
                continue
            df = df.rename(columns={"zip code": "zip5"})
            df = df[df["zip5"].isin(nm_zip_set)].copy()
            df["year"] = year
            df["sector"] = sector
            all_rows.append(df[["zip5", "year", "sector", "ESTAB"]])
            time.sleep(1)

    panel = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(TBL, "nm_zip_zbp_raw.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}, {len(panel)} rows")
    print(panel.groupby("sector").size())


if __name__ == "__main__":
    main()
