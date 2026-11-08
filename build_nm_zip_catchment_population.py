"""
Compute "population within 70 miles" for each NM dispensary ZIP code, per
explicit user request -- a catchment/gravity population control distinct
from the ZIP's own resident population, using ALL US ZIP codes (not just
NM's), since the whole point is to pick up nearby out-of-state population
(e.g. El Paso, TX inflating the catchment population of Dona Ana border
ZIPs) that a same-state-only population figure would miss entirely.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
RADIUS_MI = 70
EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def main():
    allzips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    allzips = allzips.dropna(subset=["lat", "lng", "population"])
    print(f"National ZIP reference: {len(allzips)} ZIP codes with population")

    nm = pd.read_csv(os.path.join(TBL, "nm_zip_border_classification.csv"), dtype={"zip5": str})
    nm = nm.dropna(subset=["zip_lat", "zip_lon"]).copy()

    all_lat = allzips["lat"].to_numpy()
    all_lon = allzips["lng"].to_numpy()
    all_pop = allzips["population"].to_numpy()

    catchment = np.zeros(len(nm))
    n_zips_in_radius = np.zeros(len(nm), dtype=int)
    for i, (lat, lon) in enumerate(zip(nm["zip_lat"].to_numpy(), nm["zip_lon"].to_numpy())):
        d = haversine_miles(lat, lon, all_lat, all_lon)
        mask = d <= RADIUS_MI
        catchment[i] = all_pop[mask].sum()
        n_zips_in_radius[i] = mask.sum()

    nm["population_within_70mi"] = catchment
    nm["n_zips_within_70mi"] = n_zips_in_radius

    out = os.path.join(TBL, "nm_zip_catchment_population.csv")
    nm[["zip5", "zip_population", "population_within_70mi", "n_zips_within_70mi",
        "border_zip", "dist_nearest_nonlegal_city_mi"]].to_csv(out, index=False)
    print(f"Wrote {out}: {len(nm)} ZIP codes")
    print("\nCatchment population summary, border vs interior:")
    print(nm.groupby("border_zip")["population_within_70mi"].describe())


if __name__ == "__main__":
    main()
