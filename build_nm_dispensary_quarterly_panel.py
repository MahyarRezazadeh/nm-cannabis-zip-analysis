"""
Dispensary-quarter panel for the dispensary-level event study, built from
the 29,895-row monthly revenue panel (nm_dispensary_monthly_revenue_panel.csv)
aggregated to quarters, merged with the border classification from
build_nm_dispensary_border_classification.py.

Three outcomes, matching the paper's ZIP-level design as closely as a
single-firm unit of observation allows:

1. log_revenue_percapita: log(dispensary total_sales / county population)
   -- the direct dispensary-level analog of the ZIP design's outcome.

2. log_revenue: log(dispensary total_sales) -- the ZIP design's "revenue
   per store" outcome divides ZIP-level revenue by the ZIP's store count;
   at the single-dispensary level that ratio collapses to just this
   dispensary's own revenue (store count = 1 by construction), so we
   report it directly rather than force a degenerate "per store" label.

3. local_density: (count of NM dispensaries active within 10 miles of
   this dispensary that quarter, including itself) / (population within
   10 miles of this dispensary / 1000) -- "dispensary density" at the
   ZIP/county level is a market-structure count that has no literal
   per-firm equivalent, so this substitutes the natural individual-level
   counterpart: local competitive density around this specific location,
   which (unlike a fixed market-level count) varies quarter to quarter as
   nearby dispensaries enter and exit. Flagged explicitly as a
   reinterpretation, not a strict analog, in the paper text.

Only dispensaries with valid coordinates (1,192 of 1,253) are included,
since both the border classification and local_density require them.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
EARTH_RADIUS_MI = 3958.8
LOCAL_RADIUS_MI = 10


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def main():
    panel = pd.read_csv(os.path.join(TBL, "nm_dispensary_monthly_revenue_panel.csv"),
                         dtype={"fips5": str, "county_fips": str})
    panel = panel.dropna(subset=["lat", "lon"]).copy()
    panel["quarter"] = ((panel["month"] - 1) // 3 + 1).astype(int)
    panel["yq"] = panel["year"].astype(str) + "Q" + panel["quarter"].astype(str)

    q = panel.groupby(["dispensary_id", "yq", "year", "quarter"]).agg(
        total_sales=("total_sales", "sum"),
        population=("population", "first"),
        fips5=("fips5", "first"),
        county_name=("county_name", "first"),
        lon=("lon", "first"),
        lat=("lat", "first"),
    ).reset_index()
    print(f"{len(q)} dispensary-quarter rows, {q.dispensary_id.nunique()} dispensaries, "
          f"{q.yq.nunique()} quarters")

    border = pd.read_csv(os.path.join(TBL, "nm_dispensary_border_classification.csv"))
    q = q.merge(border[["dispensary_id", "dist_stateline_mi", "nonlegal_pop_within_75mi",
                         "border_dispensary"]], on="dispensary_id", how="inner")
    print(f"After merging border classification (drops unmatched coords): {len(q)} rows, "
          f"{q.dispensary_id.nunique()} dispensaries")

    # ---------- outcome 1: revenue per capita ----------
    q["log_revenue_percapita"] = np.log(q["total_sales"] / q["population"])

    # ---------- outcome 2: revenue (dispensary-level "per store" analog) ----------
    q["log_revenue"] = np.log(q["total_sales"].clip(lower=1))

    # ---------- outcome 3: local competitive density ----------
    # population within 10mi of each dispensary (fixed per dispensary, national ZIP reference)
    allzips = pd.read_csv(r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana\data\raw\uszips_simplemaps.csv",
                           dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    z_lat, z_lon, z_pop = allzips["lat"].to_numpy(), allzips["lng"].to_numpy(), allzips["population"].to_numpy()

    uniq_disp = q.drop_duplicates("dispensary_id")[["dispensary_id", "lat", "lon"]].reset_index(drop=True)
    pop_within10 = np.zeros(len(uniq_disp))
    for i, (lat, lon) in enumerate(zip(uniq_disp["lat"].to_numpy(), uniq_disp["lon"].to_numpy())):
        d = haversine_miles(lat, lon, z_lat, z_lon)
        pop_within10[i] = z_pop[d <= LOCAL_RADIUS_MI].sum()
    uniq_disp["pop_within_10mi"] = pop_within10
    q = q.merge(uniq_disp[["dispensary_id", "pop_within_10mi"]], on="dispensary_id", how="left")

    # neighbor count: for each quarter, how many (other) geocoded NM dispensaries are
    # active (appear that quarter) within 10mi -- computed per quarter since the set of
    # active dispensaries changes as stores enter/exit
    dlat = q["lat"].to_numpy()
    dlon = q["lon"].to_numpy()
    neighbor_count = np.zeros(len(q), dtype=int)
    for yq_val, idx in q.groupby("yq").indices.items():
        idx = np.asarray(idx)
        lat_q, lon_q = dlat[idx], dlon[idx]
        for j, i_row in enumerate(idx):
            d = haversine_miles(lat_q[j], lon_q[j], lat_q, lon_q)
            neighbor_count[i_row] = int((d <= LOCAL_RADIUS_MI).sum())  # includes self

    q["local_dispensary_count"] = neighbor_count
    q["local_density"] = q["local_dispensary_count"] / (q["pop_within_10mi"] / 1000)
    q["log_local_density"] = np.log(q["local_density"].replace(0, np.nan))

    # ---------- calendar-quarter period index (t=0 at panel start, matches ZIP design) ----------
    periods = sorted(q["yq"].unique(), key=lambda s: (int(s[:4]), int(s[-1])))
    period_map = {p: i for i, p in enumerate(periods)}
    q["period"] = q["yq"].map(period_map)
    print(f"\nPanel spans {periods[0]} (period 0) to {periods[-1]} (period {len(periods) - 1})")

    print(f"\nBorder dispensary-quarters: {q[q.border_dispensary == 1].shape[0]}, "
          f"Interior: {q[q.border_dispensary == 0].shape[0]}")
    print(f"Unique border dispensaries: {q[q.border_dispensary == 1].dispensary_id.nunique()}, "
          f"Unique interior dispensaries: {q[q.border_dispensary == 0].dispensary_id.nunique()}")

    out_path = os.path.join(TBL, "nm_dispensary_quarterly_panel.csv")
    q.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    dta_path = os.path.join(TBL, "nm_dispensary_quarterly_panel.dta")
    q_dta = q.copy()
    q_dta = q_dta.replace([np.inf, -np.inf], np.nan)
    str_cols = q_dta.select_dtypes(include="object").columns
    q_dta[str_cols] = q_dta[str_cols].fillna("")
    q_dta.to_stata(dta_path, write_index=False, version=118)
    print(f"Wrote {dta_path}")


if __name__ == "__main__":
    main()
