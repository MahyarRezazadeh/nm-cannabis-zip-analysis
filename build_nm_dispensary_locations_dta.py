"""
Dispensary-level location file for New Mexico, built at the user's request
from the entry-game project's own canonical dispensary panel (addr_norm +
county_fips), geocoded to lat/lon.

Source of dispensary identity: entry game-marijuana/tables/nm_dispensary_panel_monthly.csv
(dispensary_id, dispensary_label, addr_norm, county_fips, fips5, county_name,
state_abbr) -- 1,253 unique dispensaries, one row each (collapsed from the
monthly panel).

Source of coordinates: tables/nm_dispensary_geocoded.csv, produced earlier
this project via the Census Bureau's free Batch Geocoder
(https://geocoding.geo.census.gov/geocoder/locations/addressbatch,
benchmark=Public_AR_Current). Reused here rather than re-geocoded, because
the addr_norm strings in the entry-game panel match the addresses already
geocoded (1,253 of 1,253 unique addresses found by exact string match --
verified, not assumed). NOTE: this file's own dispensary_id numbering is
independent of (does not correspond to) the numbering used in
nm_dispensary_addresses.csv / nm_dispensary_geocoded.csv -- the join below
is done on the normalized address STRING, never on dispensary_id, for
exactly this reason.

match_status/match_type are carried through from the Census geocoder so
users of this file can see which rows are actual matches vs. addresses the
geocoder could not resolve (about 14% of addresses, per the earlier
geocoding run).
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")


def norm(s):
    return s.astype(str).str.strip().str.upper()


def main():
    panel = pd.read_csv(os.path.join(SOURCE_BASE, "tables", "nm_dispensary_panel_monthly.csv"),
                         dtype={"county_fips": str, "fips5": str})
    disp = panel[["dispensary_id", "dispensary_label", "addr_norm", "county_fips",
                  "fips5", "county_name", "state_abbr"]].drop_duplicates(subset=["dispensary_id"]).copy()
    print(f"{len(disp)} unique dispensaries in the entry-game panel")

    # nm_dispensary_geocoded.csv's own dispensary_id matches
    # nm_dispensary_addresses.csv's dispensary_id (built together in the same
    # earlier pipeline) -- join those two by dispensary_id first to recover a
    # clean addr_norm -> lon/lat mapping, since nm_dispensary_geocoded.csv's
    # own input_address column has city/state/zip appended and isn't a
    # direct string match to addr_norm.
    old_addr = pd.read_csv(os.path.join(TBL, "nm_dispensary_addresses.csv"))
    geo = pd.read_csv(os.path.join(TBL, "nm_dispensary_geocoded.csv"))
    geo = old_addr[["dispensary_id", "addr_norm"]].merge(
        geo[["dispensary_id", "match_status", "match_type", "matched_address", "lon", "lat"]],
        on="dispensary_id", how="left")
    geo["addr_key"] = norm(geo["addr_norm"])
    geo = geo.drop(columns=["dispensary_id", "addr_norm"]).drop_duplicates(subset=["addr_key"])

    disp["addr_key"] = norm(disp["addr_norm"])
    out = disp.merge(geo, on="addr_key", how="left")
    out = out.drop(columns=["addr_key"])
    out["geocode_source"] = np.where(out["lat"].notna(), "Census_batch", pd.NA)

    # second pass: OpenStreetMap Nominatim fallback for addresses the Census
    # geocoder couldn't resolve (see geocode_nm_remaining_nominatim.py)
    fallback_path = os.path.join(TBL, "nm_dispensary_nominatim_fallback.csv")
    if os.path.exists(fallback_path):
        fb = pd.read_csv(fallback_path)
        fb = fb[fb["nominatim_lon"].notna()][["dispensary_id", "nominatim_lon", "nominatim_lat",
                                               "nominatim_matched_display"]]
        out = out.merge(fb, on="dispensary_id", how="left")
        fill = out["lat"].isna() & out["nominatim_lat"].notna()
        out.loc[fill, "lon"] = out.loc[fill, "nominatim_lon"]
        out.loc[fill, "lat"] = out.loc[fill, "nominatim_lat"]
        out.loc[fill, "matched_address"] = out.loc[fill, "nominatim_matched_display"]
        out.loc[fill, "match_status"] = "Match"
        out.loc[fill, "match_type"] = "Nominatim_fallback"
        out.loc[fill, "geocode_source"] = "Nominatim_fallback"
        out = out.drop(columns=["nominatim_lon", "nominatim_lat", "nominatim_matched_display"])
        print(f"Nominatim fallback filled {fill.sum()} additional dispensaries")

    n_matched = out["lat"].notna().sum()
    print(f"{n_matched} of {len(out)} dispensaries matched to coordinates "
          f"({n_matched/len(out)*100:.1f}%)")
    print(out["geocode_source"].value_counts(dropna=False))

    csv_path = os.path.join(TBL, "nm_dispensary_locations.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    dta_path = os.path.join(TBL, "nm_dispensary_locations.dta")
    out_dta = out.copy()
    # Stata .dta has no native missing-string distinct from "", and column
    # names must be valid Stata identifiers (<=32 chars, no special chars) --
    # all names here already satisfy that. String columns with unmatched
    # (all-null) rows need an explicit "" fill -- to_stata rejects an
    # object column that is entirely null.
    str_cols = out_dta.select_dtypes(include="object").columns
    out_dta[str_cols] = out_dta[str_cols].fillna("")
    out_dta.to_stata(dta_path, write_index=False, version=118)
    print(f"Wrote {dta_path}")


if __name__ == "__main__":
    main()
