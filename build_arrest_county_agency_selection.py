"""
Select the county sample and primary reporting agency (county sheriff,
falling back to the largest available city PD) for the arrest-based
demand-spillover panel. Scoped down from the full 360-county universe to
keep the FBI CDE API pull tractable (each agency-offense pull is one live
API call, and the API has shown real rate-limit sensitivity):
  - all 47 border counties (the treatment units -- needed in full)
  - all interior counties in OK (75) and UT (20) -- small states, cheap
  - the 80 FARTHEST interior counties in TX (of 218), ranked by distance
    to the nearest legalizing neighbor, as clean, unambiguous controls
    (avoids picking near-threshold TX counties that are barely outside
    the 75-mile border cutoff)
"""
import os
import re
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
SCRATCH = r"C:\Users\mahya\AppData\Local\Temp\claude\C--Users-mahya\b58dba2e-28c6-4dc5-b9da-8c0cf751f866\scratchpad"

N_TX_INTERIOR = 80


def normalize_county(name):
    name = str(name).upper()
    name = re.sub(r"\s+COUNTY$", "", name)
    return name.strip()


def main():
    cls = pd.read_csv(os.path.join(TBL, "arrest_county_border_classification.csv"), dtype={"fips5": str})
    cls["county_key"] = cls["county_name"].apply(normalize_county)

    agents = pd.read_csv(os.path.join(SCRATCH, "primary_agency_per_county.csv"))

    merged = cls.merge(agents, on=["state_abbr", "county_key"], how="left")
    missing = merged[merged.ori.isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} counties have no matched agency:")
        print(missing[["state_abbr", "county_name"]].to_string(index=False))

    merged = merged.dropna(subset=["ori"]).copy()

    border = merged[merged.border_county == 1]
    ok_interior = merged[(merged.state_abbr == "OK") & (merged.border_county == 0)]
    ut_interior = merged[(merged.state_abbr == "UT") & (merged.border_county == 0)]
    tx_interior_all = merged[(merged.state_abbr == "TX") & (merged.border_county == 0)]
    tx_interior = tx_interior_all.sort_values("dist_nearest_legal_mi", ascending=False).head(N_TX_INTERIOR)

    selected = pd.concat([border, ok_interior, ut_interior, tx_interior], ignore_index=True)
    selected["never_treated"] = (selected["border_county"] == 0).astype(int)

    print(f"\nSelected {len(selected)} counties total:")
    print(selected.groupby(["state_abbr", "border_county"]).size())

    out_path = os.path.join(TBL, "arrest_county_agency_selection.csv")
    selected.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
