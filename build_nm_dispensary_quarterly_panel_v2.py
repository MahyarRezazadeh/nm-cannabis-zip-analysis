"""
v2 of the dispensary-quarter panel, addressing four noise sources
diagnosed after the first pass (see _diagnose_dispensary_noise.py):

1. local_density's 10-mile radius produced a badly right-skewed ratio
   (94 dispensaries with <500 people in that radius, max ratio 97.1) --
   widened to 25 miles, which pushes the population denominator well
   above the danger zone for nearly all dispensaries.

2. Chain/brand fixed effect, extracted mechanically from dispensary_label
   (split on the first "-" or "/", strip legal-entity suffixes). 202
   chains have >=2 locations (746 of 1,253 dispensaries); the remaining
   507 are effectively singleton brands. Only chains with a genuine mix
   of border and interior locations contribute identifying variation for
   the border coefficient once brand FE is added -- checked explicitly
   in the analysis script via matrix rank, not assumed.

4. County population is a poor, overly coarse proxy for a single
   dispensary's local market (two dispensaries in the same huge county
   can face very different real local demand). Replaced with the same
   25-mile-radius population used for local_density's denominator, for
   ALL three outcomes' population control, for consistency.

5. dispensary_age_q (quarters since this dispensary's own entry_year/
   entry_month) is added so the analysis script can down-weight
   brand-new, still-ramping-up dispensary-quarters via WLS.

Also assigns each dispensary to its nearest NM ZIP centroid (national
ZIP reference file) so the analysis script can cluster standard errors
at the ZIP level instead of the county level, per explicit request.
"""
import os
import re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
EARTH_RADIUS_MI = 3958.8
LOCAL_RADIUS_MI = 25


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def extract_brand(label):
    s = str(label).upper()
    s = re.split(r"\s*[-/]\s*", s)[0]
    s = re.sub(r"\bDBA\b.*", "", s)
    s = re.sub(r"\b(LLC|L\.L\.C\.|INC|CORP|CO|LTD|LP)\.?\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    panel = pd.read_csv(os.path.join(TBL, "nm_dispensary_monthly_revenue_panel.csv"),
                         dtype={"fips5": str, "county_fips": str})
    panel = panel.dropna(subset=["lat", "lon"]).copy()
    panel["quarter"] = ((panel["month"] - 1) // 3 + 1).astype(int)
    panel["yq"] = panel["year"].astype(str) + "Q" + panel["quarter"].astype(str)
    panel["brand"] = panel["dispensary_label"].apply(extract_brand)

    q = panel.groupby(["dispensary_id", "yq", "year", "quarter"]).agg(
        total_sales=("total_sales", "sum"),
        population=("population", "first"),
        fips5=("fips5", "first"),
        county_name=("county_name", "first"),
        lon=("lon", "first"),
        lat=("lat", "first"),
        brand=("brand", "first"),
        entry_year=("entry_year", "first"),
        entry_month=("entry_month", "first"),
    ).reset_index()
    print(f"{len(q)} dispensary-quarter rows, {q.dispensary_id.nunique()} dispensaries, "
          f"{q.yq.nunique()} quarters")

    border = pd.read_csv(os.path.join(TBL, "nm_dispensary_border_classification.csv"))
    q = q.merge(border[["dispensary_id", "dist_stateline_mi", "nonlegal_pop_within_75mi",
                         "border_dispensary"]], on="dispensary_id", how="inner")
    print(f"After merging border classification: {len(q)} rows, {q.dispensary_id.nunique()} dispensaries")

    q["log_revenue_percapita"] = np.log(q["total_sales"] / q["population"])
    q["log_revenue"] = np.log(q["total_sales"].clip(lower=1))

    SOURCE_ZIPS = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana\data\raw\uszips_simplemaps.csv"
    allzips = pd.read_csv(SOURCE_ZIPS, dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    z_lat, z_lon, z_pop = allzips["lat"].to_numpy(), allzips["lng"].to_numpy(), allzips["population"].to_numpy()

    uniq_disp = q.drop_duplicates("dispensary_id")[["dispensary_id", "lat", "lon"]].reset_index(drop=True)

    # ---------- local population within 25mi (fixed per dispensary) ----------
    pop_within = np.zeros(len(uniq_disp))
    for i, (lat, lon) in enumerate(zip(uniq_disp["lat"].to_numpy(), uniq_disp["lon"].to_numpy())):
        d = haversine_miles(lat, lon, z_lat, z_lon)
        pop_within[i] = z_pop[d <= LOCAL_RADIUS_MI].sum()
    uniq_disp["pop_within_25mi"] = pop_within

    # ---------- nearest NM ZIP (for zip-level clustering) ----------
    nm_zips = allzips[allzips["state_id"] == "NM"].reset_index(drop=True)
    nz_lat, nz_lon, nz5 = (nm_zips["lat"].to_numpy(), nm_zips["lng"].to_numpy(),
                           nm_zips["zip"].to_numpy())
    nearest_zip = []
    for lat, lon in zip(uniq_disp["lat"].to_numpy(), uniq_disp["lon"].to_numpy()):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        nearest_zip.append(nz5[np.argmin(d)])
    uniq_disp["zip5_nearest"] = nearest_zip

    q = q.merge(uniq_disp[["dispensary_id", "pop_within_25mi", "zip5_nearest"]], on="dispensary_id", how="left")
    q["log_local_population"] = np.log(q["pop_within_25mi"])

    # ---------- neighbor count within 25mi, per quarter (entry/exit-aware) ----------
    dlat = q["lat"].to_numpy()
    dlon = q["lon"].to_numpy()
    neighbor_count = np.zeros(len(q), dtype=int)
    for yq_val, idx in q.groupby("yq").indices.items():
        idx = np.asarray(idx)
        lat_q, lon_q = dlat[idx], dlon[idx]
        for j, i_row in enumerate(idx):
            d = haversine_miles(lat_q[j], lon_q[j], lat_q, lon_q)
            neighbor_count[i_row] = int((d <= LOCAL_RADIUS_MI).sum())

    q["local_dispensary_count_25mi"] = neighbor_count
    q["local_density"] = q["local_dispensary_count_25mi"] / (q["pop_within_25mi"] / 1000)
    q["log_local_density"] = np.log(q["local_density"].replace(0, np.nan))

    # ---------- dispensary age in quarters (for WLS weighting) ----------
    periods = sorted(q["yq"].unique(), key=lambda s: (int(s[:4]), int(s[-1])))
    period_map = {p: i for i, p in enumerate(periods)}
    q["period"] = q["yq"].map(period_map)
    entry_period_idx = q["entry_year"].astype(str) + "Q" + (((q["entry_month"] - 1) // 3) + 1).astype(str)
    q["entry_period"] = entry_period_idx.map(period_map)
    q["dispensary_age_q"] = (q["period"] - q["entry_period"]).clip(lower=0)

    print(f"\nPanel spans {periods[0]} (period 0) to {periods[-1]} (period {len(periods) - 1})")
    print(f"Border dispensary-quarters: {q[q.border_dispensary == 1].shape[0]}, "
          f"Interior: {q[q.border_dispensary == 0].shape[0]}")
    print(f"Unique ZIPs (nearest-match, for clustering): {q.zip5_nearest.nunique()}")
    print(f"Dispensary age (quarters): median={q.dispensary_age_q.median():.0f}, "
          f"max={q.dispensary_age_q.max():.0f}")

    out_path = os.path.join(TBL, "nm_dispensary_quarterly_panel_v2.csv")
    q.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    dta_path = os.path.join(TBL, "nm_dispensary_quarterly_panel_v2.dta")
    q_dta = q.replace([np.inf, -np.inf], np.nan)
    str_cols = q_dta.select_dtypes(include="object").columns
    q_dta[str_cols] = q_dta[str_cols].fillna("")
    q_dta.to_stata(dta_path, write_index=False, version=118)
    print(f"Wrote {dta_path}")


if __name__ == "__main__":
    main()
