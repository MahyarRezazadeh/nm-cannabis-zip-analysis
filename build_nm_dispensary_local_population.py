"""
Population within a small radius of each New Mexico dispensary, from 2020
Census block population (the finest official geography).

Why not reuse the panel's pop_within_25mi machinery: that sums ZIP-CENTROID
population within a radius (build_nm_dispensary_quarterly_panel_v2.py). NM ZIP
codes are 3-80 km across, so at a 1 km radius the ZIP-centroid method returns
0 (no centroid inside) or a whole ZIP's population (one centroid inside) --
meaningless. Census blocks (median area well under 1 km^2 in populated areas)
are the right unit.

Source: TIGER/Line 2020 tabblock20 for New Mexico (state FIPS 35),
data/tl_2020_35_tabblock20/, attribute POP20 = 2020 Census resident population.
107,215 blocks; ~40k have POP20 > 0 (only those can contribute, so the rest
are dropped -- exact, not an approximation).

Two measures per radius, per dispensary:
  *_areal    : area-apportioned -- sum over blocks intersecting the buffer of
               POP20 * (intersected area / full block area). Handles blocks
               that straddle the buffer edge. This is the preferred measure.
  *_centroid : sum of POP20 over blocks whose Census internal point falls
               inside the buffer. Coarser; reported for comparison.

Dispensary coordinates: nm_dispensary_quarterly_panel_v3.csv (1,192 geocoded
dispensaries; 607 exact + 475 street-interpolated Census matches + 173
Nominatim fallback -- interpolated points carry ~50-150 m error, which matters
at a 1 km radius, so treat the 1 km numbers as approximate).

Projected CRS: EPSG:26913 (NAD83 / UTM zone 13N, metres) -- appropriate for NM.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
BLOCKS_SHP = os.path.join(BASE, "data", "tl_2020_35_tabblock20", "tl_2020_35_tabblock20.shp")
METRIC_CRS = "EPSG:26913"

RADII_M = {
    "0_5km": 500.0,
    "1km": 1000.0,
    "2km": 2000.0,
    "3km": 3000.0,
    "5km": 5000.0,
}


def main():
    # ---- blocks ----
    blocks = gpd.read_file(BLOCKS_SHP, columns=["GEOID20", "POP20", "INTPTLAT20", "INTPTLON20"])
    blocks = blocks[blocks["POP20"] > 0].copy()
    blocks = blocks.to_crs(METRIC_CRS)
    blocks["block_area_m2"] = blocks.geometry.area
    blk_pts = gpd.GeoDataFrame(
        blocks[["GEOID20", "POP20"]].copy(),
        geometry=gpd.points_from_xy(blocks["INTPTLON20"].astype(float), blocks["INTPTLAT20"].astype(float)),
        crs="EPSG:4269",
    ).to_crs(METRIC_CRS)
    print(f"{len(blocks):,} populated NM blocks (POP20 > 0); total POP20 = {blocks['POP20'].sum():,}")

    # ---- dispensaries (unique) ----
    v3 = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"))
    disp = v3.drop_duplicates("dispensary_id")[
        ["dispensary_id", "lat", "lon", "border_dispensary", "county_name", "brand"]
    ].reset_index(drop=True)
    gdisp = gpd.GeoDataFrame(
        disp.copy(),
        geometry=gpd.points_from_xy(disp["lon"], disp["lat"]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)
    print(f"{len(gdisp):,} unique geocoded dispensaries")

    out = disp[["dispensary_id", "lat", "lon", "border_dispensary", "county_name", "brand"]].copy()

    for tag, r in RADII_M.items():
        buf = gdisp[["dispensary_id", "geometry"]].copy()
        buf["geometry"] = buf.geometry.buffer(r)

        # ---- areal apportionment ----
        inter = gpd.overlay(buf, blocks[["GEOID20", "POP20", "block_area_m2", "geometry"]],
                            how="intersection", keep_geom_type=True)
        inter["frac"] = (inter.geometry.area / inter["block_area_m2"]).clip(upper=1.0)
        inter["pop_alloc"] = inter["POP20"] * inter["frac"]
        areal = inter.groupby("dispensary_id")["pop_alloc"].sum()
        out[f"pop_{tag}_areal"] = out["dispensary_id"].map(areal).fillna(0.0).round(1)

        # ---- centroid membership ----
        cen = gpd.sjoin(blk_pts[["POP20", "geometry"]], buf, predicate="within", how="inner")
        cent = cen.groupby("dispensary_id")["POP20"].sum()
        out[f"pop_{tag}_centroid"] = out["dispensary_id"].map(cent).fillna(0).astype(int)

        print(f"  r={r:>6.0f} m ({tag}): areal mean={out[f'pop_{tag}_areal'].mean():,.0f} "
              f"median={out[f'pop_{tag}_areal'].median():,.0f} "
              f"zero={(out[f'pop_{tag}_areal'] == 0).sum()}")

    path = os.path.join(TBL, "nm_dispensary_local_population.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")

    # quick border vs interior comparison at 1 km
    for m in ["pop_1km_areal", "pop_1km_centroid", "pop_2km_areal", "pop_5km_areal"]:
        b = out.loc[out.border_dispensary == 1, m]
        i = out.loc[out.border_dispensary == 0, m]
        print(f"{m:22s}  border mean={b.mean():10,.0f}  interior mean={i.mean():10,.0f}  "
              f"border median={b.median():9,.0f}  interior median={i.median():9,.0f}")


if __name__ == "__main__":
    main()
