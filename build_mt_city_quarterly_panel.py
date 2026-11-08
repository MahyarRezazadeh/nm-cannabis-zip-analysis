"""
Montana city-level dispensary-density panel, built from
mt_dispensary_locations.csv (licensee, city -- no street address, no
dates). This is necessarily a STATIC SNAPSHOT repeated across every
quarter in MT's existing county-panel window (2022Q1-2025Q4), the same
already-disclosed limitation as MT's county-level dispensary counts
elsewhere in this project -- there is no per-dispensary date information
available to build genuine time variation. No revenue outcome: MT's
sales data (montana_county_month_sales.csv) is county-level only.

City coordinates/population: population-weighted centroid and summed
population of each city's own ZIP codes, from the national ZIP reference.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")

QUARTER_START = (2022, 1)
QUARTER_END = (2025, 4)


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
    disp = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "mt_dispensary_locations.csv"))
    disp["city_key"] = disp["city"].str.strip().str.upper()
    print(f"{len(disp)} MT dispensary records, {disp.city_key.nunique()} distinct cities")

    city_counts = disp.groupby("city_key").size().rename("dispensary_count").reset_index()

    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    mt_zips = allzips[allzips.state_id == "MT"].dropna(subset=["lat", "lng", "population"])
    mt_zips["city_key"] = mt_zips["city"].str.upper()
    city_pop = mt_zips.groupby("city_key").apply(
        lambda g: pd.Series({
            "population": g["population"].sum(),
            "lat": np.average(g["lat"], weights=g["population"].clip(lower=1)),
            "lon": np.average(g["lng"], weights=g["population"].clip(lower=1)),
        })
    ).reset_index()

    panel_cs = city_counts.merge(city_pop, on="city_key", how="left")
    n_before = len(panel_cs)
    panel_cs = panel_cs.dropna(subset=["population"])
    print(f"City population matched: {len(panel_cs)} of {n_before} cities "
          f"({panel_cs.dispensary_count.sum()} of {city_counts.dispensary_count.sum()} dispensaries)")

    panel_cs["density_per_1000"] = 1000 * panel_cs["dispensary_count"] / panel_cs["population"]
    panel_cs["log_density"] = np.log(panel_cs["density_per_1000"])
    panel_cs = panel_cs.rename(columns={"city_key": "city"})

    quarters = quarters_between(QUARTER_START, QUARTER_END)
    q_df = pd.DataFrame(quarters, columns=["year", "quarter"])
    q_df["key"] = 1
    panel_cs["key"] = 1
    panel = panel_cs.merge(q_df, on="key").drop(columns="key")
    panel["yq"] = panel["year"].astype(str) + "Q" + panel["quarter"].astype(str)

    out_path = os.path.join(TBL, "mt_city_quarterly_panel.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"{panel.city.nunique()} cities x {len(quarters)} quarters = {len(panel)} rows "
          f"({panel.yq.min()} to {panel.yq.max()})")


if __name__ == "__main__":
    main()
