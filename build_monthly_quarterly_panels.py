"""
Build county-month and county-quarter panels for CO, OR, MT, NM (Washington
dropped per explicit user request). Reuses the exact monthly-frequency raw
sources already assembled for the .dta export
(export_state_panels_dta.py) -- CO's own DOR file (county-month, with real
monthly retailer counts), OR's raw OLCC transaction pull (aggregated to
county-month), MT's scraped county-month sales, and NM's dispensary-level
monthly panel (aggregated to county-month, with a REAL monthly dispensary
count since every store's presence is directly observed each month).

Time-varying dispensary counts only exist for CO and NM. OR and MT's
per-store data has only ever existed as a single current-year cross-section
in this project (their license registries carry no historical count), so
their monthly/quarterly "dispensary_count" here is that one snapshot value
repeated across every period -- flagged with count_is_snapshot=1. Revenue
per capita is unaffected by this and is genuine for all 4 states; revenue
per store and dispensary density should be read as CO/NM-only outcomes,
per the user's "to the extent possible" instruction.

Border status (adjacency definition) and entry_capped are state-year-level
by construction (Section timevarying / regime), so they are simply
broadcast to every month/quarter within a year -- they do not vary within
a year in this project's design.

This copy lives in the "New Mexico project" folder (moved here per explicit
user request); it reads its raw/intermediate inputs from the original
entry-game-marijuana project (SOURCE_BASE below), which still holds all raw
data, and writes its own outputs into THIS folder's tables/ subfolder --
avoiding a full duplicate of entry-game-marijuana's raw-data tree.
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
SRC_TBL = os.path.join(SOURCE_BASE, "tables")
TBL = os.path.join(BASE, "tables")
STATES = ["CO", "OR", "MT", "NM"]


def county_fips_crosswalk():
    m = pd.read_csv(os.path.join(SRC_TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    m = m[["fips5", "state_abbr", "county_name"]].drop_duplicates()
    m["county_key"] = m["county_name"].str.replace(" County", "", regex=False).str.strip().str.upper()
    return m[["fips5", "state_abbr", "county_key"]]


def co_monthly(xwalk):
    raw = pd.read_csv(os.path.join(SOURCE_BASE, "border revenue", "data", "raw", "co_marijuana_sales_by_county.csv"))
    raw = raw.dropna(subset=["year", "month", "county"])
    raw["year"] = raw["year"].astype(int)
    raw["month"] = raw["month"].astype(int)
    raw["med_sales"] = pd.to_numeric(raw["med_sales"], errors="coerce").fillna(0)
    raw["rec_sales"] = pd.to_numeric(raw["rec_sales"], errors="coerce").fillna(0)
    raw["total_sales"] = raw["med_sales"] + raw["rec_sales"]
    raw["dispensary_count"] = pd.to_numeric(raw["numberofretailmarijuanaretailers"], errors="coerce")
    raw["county_key"] = raw["county"].str.strip().str.upper()
    xw = xwalk[xwalk.state_abbr == "CO"]
    df = raw.merge(xw, on="county_key", how="inner")
    df["state_abbr"] = "CO"
    df["count_is_snapshot"] = 0
    return df[["fips5", "state_abbr", "year", "month", "total_sales", "dispensary_count", "count_is_snapshot"]]


def or_monthly(xwalk):
    raw = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "oregon_revenue", "oregon_olcc_raw.csv"))
    raw["date"] = pd.to_datetime(raw["date"])
    raw["year"] = raw["date"].dt.year
    raw["month"] = raw["date"].dt.month
    raw["sales"] = pd.to_numeric(raw["sales"], errors="coerce").fillna(0)
    monthly = raw.groupby(["county", "year", "month"], as_index=False)["sales"].sum()
    monthly = monthly.rename(columns={"sales": "total_sales"})
    monthly["county_key"] = monthly["county"].str.strip().str.upper()
    xw = xwalk[xwalk.state_abbr == "OR"]
    df = monthly.merge(xw, on="county_key", how="inner")
    df["state_abbr"] = "OR"

    snap = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "oregon_revenue", "oregon_active_retailers_by_county.csv"))
    snap["county_key"] = snap["county_key"].str.strip().str.upper()
    snap = snap.rename(columns={"n": "dispensary_count"})[["county_key", "dispensary_count"]]
    df = df.merge(snap, on="county_key", how="left")
    df["count_is_snapshot"] = 1
    return df[["fips5", "state_abbr", "year", "month", "total_sales", "dispensary_count", "count_is_snapshot"]]


def mt_monthly(xwalk):
    raw = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "montana_revenue", "montana_county_month_sales_clean.csv"))
    raw = raw.rename(columns={"sales": "total_sales"})
    raw["county_key"] = raw["county"].str.strip().str.upper()
    xw = xwalk[xwalk.state_abbr == "MT"]
    df = raw.merge(xw, on="county_key", how="inner")
    df["state_abbr"] = "MT"

    cross = pd.read_csv(os.path.join(SRC_TBL, "montana_revenue_perstore_cross.csv"), dtype={"fips5": str})
    snap = cross[["fips5", "dispensary_count"]].drop_duplicates()
    df = df.merge(snap, on="fips5", how="left")
    df["count_is_snapshot"] = 1
    return df[["fips5", "state_abbr", "year", "month", "total_sales", "dispensary_count", "count_is_snapshot"]]


def nm_monthly():
    m = pd.read_csv(os.path.join(SRC_TBL, "nm_dispensary_panel_monthly.csv"), dtype={"fips5": str})
    grp = m.groupby(["fips5", "year", "month"]).agg(
        total_sales=("total_sales", "sum"),
        dispensary_count=("dispensary_id", "nunique")).reset_index()
    grp["state_abbr"] = "NM"
    grp["count_is_snapshot"] = 0
    return grp[["fips5", "state_abbr", "year", "month", "total_sales", "dispensary_count", "count_is_snapshot"]]


def attach_yearlevel_covariates(df):
    m = pd.read_csv(os.path.join(SRC_TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    m = m[["fips5", "year", "population", "entry_capped", "county_name"]].drop_duplicates()
    adj = pd.read_csv(os.path.join(SRC_TBL, "border_adjacent_by_year.csv"), dtype={"fips5": str})[
        ["fips5", "year", "border_adjacent"]]
    df = df.merge(m, on=["fips5", "year"], how="left")
    df = df.merge(adj, on=["fips5", "year"], how="left")
    return df


def main():
    xw = county_fips_crosswalk()
    parts = [co_monthly(xw), or_monthly(xw), mt_monthly(xw), nm_monthly()]
    monthly = pd.concat(parts, ignore_index=True)
    monthly = monthly.dropna(subset=["fips5"]).copy()
    monthly["fips5"] = monthly["fips5"].astype(str).str.zfill(5)
    monthly = attach_yearlevel_covariates(monthly)

    before = len(monthly)
    monthly = monthly[monthly["population"].notna() & monthly["border_adjacent"].notna()].copy()
    print(f"Monthly panel: {before} raw rows -> {len(monthly)} after dropping unmatched county-years "
          f"(missing population/border_adjacent merge)")

    monthly["revenue_per_capita"] = monthly["total_sales"] / monthly["population"]
    monthly["revenue_per_store"] = monthly["total_sales"] / monthly["dispensary_count"]
    monthly["dispensary_per_1000"] = 1000 * monthly["dispensary_count"] / monthly["population"]
    for col in ["revenue_per_capita", "revenue_per_store", "dispensary_per_1000"]:
        monthly[f"log_{col}"] = np.log(monthly[col].where(monthly[col] > 0))
    monthly["log_population"] = np.log(monthly["population"])
    monthly["yq"] = monthly["year"].astype(str) + "Q" + (((monthly["month"] - 1) // 3) + 1).astype(str)
    monthly["ym"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)

    monthly = monthly.sort_values(["state_abbr", "fips5", "year", "month"]).reset_index(drop=True)
    monthly.to_csv(os.path.join(TBL, "master_monthly_panel_no_wa.csv"), index=False)
    print(f"Wrote master_monthly_panel_no_wa.csv: {monthly.shape}, states: {sorted(monthly.state_abbr.unique())}, "
          f"years {monthly.year.min()}-{monthly.year.max()}")

    # ---------- quarterly aggregation ----------
    q = monthly.groupby(["fips5", "state_abbr", "county_name", "year",
                          monthly["month"].sub(1).floordiv(3).add(1).rename("quarter")]).agg(
        total_sales=("total_sales", "sum"),
        dispensary_count=("dispensary_count", "mean"),
        count_is_snapshot=("count_is_snapshot", "max"),
        population=("population", "first"),
        entry_capped=("entry_capped", "first"),
        border_adjacent=("border_adjacent", "first"),
        n_months=("total_sales", "size"),
    ).reset_index()
    q = q[q["n_months"] == 3].drop(columns="n_months")  # require complete quarters only

    q["revenue_per_capita"] = q["total_sales"] / q["population"]
    q["revenue_per_store"] = q["total_sales"] / q["dispensary_count"]
    q["dispensary_per_1000"] = 1000 * q["dispensary_count"] / q["population"]
    for col in ["revenue_per_capita", "revenue_per_store", "dispensary_per_1000"]:
        q[f"log_{col}"] = np.log(q[col].where(q[col] > 0))
    q["log_population"] = np.log(q["population"])
    q["yq"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)

    q = q.sort_values(["state_abbr", "fips5", "year", "quarter"]).reset_index(drop=True)
    q.to_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), index=False)
    print(f"Wrote master_quarterly_panel_no_wa.csv: {q.shape}, states: {sorted(q.state_abbr.unique())}, "
          f"years {q.year.min()}-{q.year.max()}")

    print("\n--- Coverage by state (monthly) ---")
    print(monthly.groupby("state_abbr").agg(
        n_rows=("fips5", "size"), n_counties=("fips5", "nunique"),
        year_min=("year", "min"), year_max=("year", "max"),
        count_is_snapshot=("count_is_snapshot", "max")))


if __name__ == "__main__":
    os.makedirs(TBL, exist_ok=True)
    main()
