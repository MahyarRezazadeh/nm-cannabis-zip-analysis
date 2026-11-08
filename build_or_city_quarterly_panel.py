"""
Oregon city-level quarterly dispensary-density panel, built from OLCC's
public "Cannabis Business Licenses and Endorsements" dataset
(data.oregon.gov resource q32u-cmam), filtered to currently-active
RECREATIONAL RETAILER licenses (license_expired == "No").

DESIGN CHOICE, disclosed rather than silently assumed: this is a CURRENT
SNAPSHOT repeated across every quarter in the panel, matching the same
already-disclosed limitation for OR's and MT's county-level dispensary
counts elsewhere in this project. We tried building a genuinely
time-varying version from each license's effective_date/expiration_date
first, but effective_date reflects the start of the CURRENT license
period (often a renewal), not a store's true original opening date --
that produced only 503 license-quarter rows from 332 licenses over 22
quarters (~1.5 quarters/license), an artifact of renewal-cycle timing,
not real entry/exit history. A static snapshot, honestly labeled as
such, is more defensible than fabricated time variation.

No revenue outcome: OR's own sales data (oregon_olcc_raw.csv) is
county-level only, with no dispensary or city identifier.

City parsing: physical_address format is "STREET CITY OR ZIP[-ZIP4]"
with no delimiter between street and city, so a naive regex can't tell
where the city name starts (e.g. "...AVE PORTLAND" vs "...AVE, MEDFORD").
Instead, match the END of the address against a whitelist of actual
Oregon city names (from the national ZIP reference), trying the longest
multi-word city name first (e.g. "LAKE OSWEGO", "OREGON CITY").
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")

QUARTER_START = (2017, 1)   # matches OR's county-panel start
QUARTER_END = (2025, 4)


def build_city_whitelist():
    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    or_zips = allzips[allzips.state_id == "OR"].dropna(subset=["lat", "lng", "population"])
    cities = sorted(or_zips["city"].str.upper().unique(), key=len, reverse=True)
    return cities, or_zips


def match_city(addr, cities_sorted):
    if not isinstance(addr, str) or "exempt" in addr.lower():
        return None, None
    s = addr.strip().upper()
    if " OR " not in s:
        return None, None
    head = s.rsplit(" OR ", 1)[0]  # everything before the trailing " OR <zip>"
    for city in cities_sorted:
        if head.endswith(city):
            return city, head[:-len(city)].strip()
    return None, None


def quarters_between(q0, q1):
    y0, qq0 = q0
    y1, qq1 = q1
    out = []
    y, q = y0, qq0
    while (y, q) <= (y1, qq1):
        out.append((y, q))
        q += 1
        if q == 5:
            q = 1
            y += 1
    return out


def main():
    with open(os.path.join(TBL, "or_retail_licenses_raw.json"), encoding="utf-8-sig") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df = df[df["license_expired"] == "No"].copy()
    print(f"{len(df)} currently-active recreational retailer licenses")

    cities_sorted, or_zips = build_city_whitelist()
    parsed = df["physical_address"].apply(lambda a: match_city(a, cities_sorted))
    df["city"] = [p[0] for p in parsed]
    df = df.dropna(subset=["city"]).copy()
    print(f"{len(df)} records matched to a known OR city ({df.city.nunique()} distinct cities)")

    city_counts = df.groupby("city").agg(dispensary_count=("license_number", "nunique")).reset_index()

    city_pop = or_zips.groupby(or_zips["city"].str.upper()).apply(
        lambda g: pd.Series({
            "population": g["population"].sum(),
            "lat": np.average(g["lat"], weights=g["population"].clip(lower=1)),
            "lon": np.average(g["lng"], weights=g["population"].clip(lower=1)),
        })
    ).reset_index().rename(columns={"index": "city"})

    panel_cs = city_counts.merge(city_pop, on="city", how="left")
    n_before = len(panel_cs)
    panel_cs = panel_cs.dropna(subset=["population"])
    print(f"City population matched: {len(panel_cs)} of {n_before} cities")

    panel_cs["density_per_1000"] = 1000 * panel_cs["dispensary_count"] / panel_cs["population"]
    panel_cs["log_density"] = np.log(panel_cs["density_per_1000"])

    quarters = quarters_between(QUARTER_START, QUARTER_END)
    q_df = pd.DataFrame(quarters, columns=["year", "quarter"])
    q_df["key"] = 1
    panel_cs["key"] = 1
    panel = panel_cs.merge(q_df, on="key").drop(columns="key")
    panel["yq"] = panel["year"].astype(str) + "Q" + panel["quarter"].astype(str)

    out_path = os.path.join(TBL, "or_city_quarterly_panel.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"{panel.city.nunique()} cities x {len(quarters)} quarters = {len(panel)} rows "
          f"({panel.yq.min()} to {panel.yq.max()})")
    print(f"Total OR dispensaries in snapshot: {panel_cs.dispensary_count.sum()}")


if __name__ == "__main__":
    main()
