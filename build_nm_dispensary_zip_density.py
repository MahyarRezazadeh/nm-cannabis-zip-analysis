"""
ZIP-scoped dispensary density, replacing the fixed-25-mile-radius
local_density with a measure matched exactly to the zip/county-level
aggregate "dispensaries per capita" construction used elsewhere in the
paper: for each dispensary-quarter, count how many geocoded dispensaries
are active THAT QUARTER in the SAME ZIP (zip5_nearest, from
build_nm_dispensary_quarterly_panel_v2.py), divided by that ZIP's
population / 1000.

This directly tests whether the null result on the fixed-radius
local_density measure was a real economic finding or a construction
mismatch (a 25-mile radius blends a lot of interior territory into any
border dispensary's "local" catchment, diluting a border/interior
contrast that the paper's zip-level analysis -- defined by administrative
boundary, not a fixed radius -- finds to be large: border ~2x interior).
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
SOURCE_ZIPS = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana\data\raw\uszips_simplemaps.csv"


def main():
    q = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v2.csv"),
                     dtype={"fips5": str, "zip5_nearest": str})

    allzips = pd.read_csv(SOURCE_ZIPS, dtype={"zip": str})
    zip_pop = allzips.set_index("zip")["population"].to_dict()
    q["zip_population"] = q["zip5_nearest"].map(zip_pop)
    print(f"ZIP population matched for {q['zip_population'].notna().sum()} of {len(q)} rows")
    print(f"ZIPs with missing population: {sorted(q[q.zip_population.isna()].zip5_nearest.unique())}")

    counts = q.groupby(["zip5_nearest", "yq"]).size().rename("n_dispensaries_zip").reset_index()
    q = q.merge(counts, on=["zip5_nearest", "yq"], how="left")

    q["density_zip"] = q["n_dispensaries_zip"] / (q["zip_population"] / 1000)
    q["log_density_zip"] = np.log(q["density_zip"].replace([0, np.inf, -np.inf], np.nan))

    print()
    print("density_zip distribution:")
    print(q["density_zip"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
    print()
    print(f"Rows with zip_population < 500: {(q.zip_population < 500).sum()}")

    out_path = os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv")
    q.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
