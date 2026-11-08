"""
Figures for the reframed Data and Background section.

  fig_nm_setting.png          -- New Mexico and its neighbors, shaded by whether the
                                 neighboring state has an operating recreational market;
                                 border vs. interior New Mexico ZIP codes and every
                                 geocoded dispensary.
  fig_nm_spatial_inequality.png -- (a) proportional-symbol map of dispensaries per 1,000
                                 residents by ZIP centroid, border vs. interior;
                                 (b) dispensaries per 1,000 residents by distance band
                                 to the nearest nonlegal-state line.
  fig1_southwest_map.png       -- Southwest-only county choropleth of dispensary density
                                 (2025), with every border county (75-mile / 1,000-resident
                                 definition) in New Mexico and Colorado outlined.

All three use the same border-county / border-ZIP definition as the paper:
centroid within 75 miles of a nonlegal-state line and >= 1,000 nonlegal-state
residents within that reach.
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
EG = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
SHP = os.path.join(EG, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")

# recreational-legal status (retail market operating during the 2022-2025 sample)
LEGAL_NB = {"CO", "AZ"}          # New Mexico's legal neighbors
NONLEGAL_NB_NM = {"TX", "OK", "UT"}
NONLEGAL_NB_CO = {"UT", "WY", "NE", "KS", "OK"}

GREEN = "#4C9F70"
GRAY = "#D9D9D9"
LIGHT = "#F4F4F4"


def load_counties():
    c = gpd.read_file(SHP)
    return c[~c["STUSPS"].isin({"AK", "HI", "PR", "VI", "GU", "AS", "MP"})].copy()


def _haversine_mi(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def adjacency_border_counties(states=("NM", "CO")):
    """
    Classify a county as a border county if its polygon shares a land border
    with a state that has no operating recreational cannabis market as of the
    sample period (retail not begun by end-2023). Counties whose only
    out-of-state neighbors are legal-market states -- Arizona, or the shared
    Colorado--New Mexico line (both legal after April 2022) -- are interior.

    Computed here from the Census county/state polygons (EPSG:5070, 500 m
    tolerance to bridge cartographic-simplification gaps) and cross-checked
    against the project's own border_adjacent_by_year.csv.

    Returns a dict {GEOID (5-digit county FIPS) -> 1/0} for counties in `states`.
    """
    counties = load_counties().to_crs(5070)
    legal = pd.read_csv(os.path.join(EG, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    legal_states = set(legal.loc[legal["retail_date"].notna()
                                 & (legal["retail_date"] <= pd.Timestamp("2023-12-31")), "state_abbr"])

    state_poly = counties.dissolve(by="STUSPS").reset_index()[["STUSPS", "geometry"]]
    state_poly["geometry"] = state_poly.geometry.buffer(500)

    out, detail = {}, {}
    for _, c in counties[counties["STUSPS"].isin(states)].iterrows():
        cg = c.geometry.buffer(500)
        touched = set(state_poly.loc[state_poly.geometry.intersects(cg)
                                     & (state_poly.STUSPS != c["STUSPS"]), "STUSPS"])
        nonlegal_touched = touched - legal_states
        out[c["GEOID"]] = int(len(nonlegal_touched) > 0)
        detail[c["GEOID"]] = (c["NAME"], sorted(touched), sorted(nonlegal_touched))

    # cross-check against border_adjacent_by_year.csv (2025 snapshot)
    try:
        ba = pd.read_csv(os.path.join(EG, "tables", "border_adjacent_by_year.csv"), dtype={"fips5": str})
        ba25 = ba[(ba.year == 2025) & (ba.state_abbr.isin(states))].set_index("fips5")["border_adjacent"].to_dict()
        mism = [f for f in out if f in ba25 and int(ba25[f]) != out[f]]
        if mism:
            print(f"  [adjacency] {len(mism)} county(ies) disagree with border_adjacent_by_year.csv:")
            for f in mism:
                print(f"     {f} {detail[f][0]}: computed touches {detail[f][1]} "
                      f"nonlegal {detail[f][2]}  (file={ba25[f]})")
    except FileNotFoundError:
        pass
    return out


# ----------------------------------------------------------------------
def fig_setting():
    apply_healy_style()
    counties = load_counties()
    show = ["NM", "CO", "AZ", "UT", "TX", "OK"]
    states = counties[counties["STUSPS"].isin(show)].dissolve(by="STUSPS").reset_index()

    zips = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv")).drop_duplicates("dispensary_id")

    fig, ax = plt.subplots(figsize=(9, 8.2))
    for _, row in states.iterrows():
        s = row["STUSPS"]
        if s == "NM":
            color = LIGHT
        elif s in LEGAL_NB:
            color = GREEN
        else:
            color = GRAY
        gpd.GeoSeries([row["geometry"]]).plot(ax=ax, color=color, edgecolor="white", linewidth=0.8)

    # NM county lines, faint
    counties[counties["STUSPS"] == "NM"].boundary.plot(ax=ax, edgecolor="#BFBFBF", linewidth=0.4)
    counties[counties["STUSPS"] == "NM"].dissolve(by="STUSPS").boundary.plot(ax=ax, edgecolor="#2B2B2B", linewidth=1.3)

    # border vs interior ZIP centroids
    b = zips[zips.border_zip == 1]
    i = zips[zips.border_zip == 0]
    ax.scatter(i["zip_lon"], i["zip_lat"], s=16, facecolors="none", edgecolors=PALETTE["blue"],
               linewidths=0.8, zorder=5, label="Interior ZIP centroid")
    ax.scatter(b["zip_lon"], b["zip_lat"], s=24, facecolors="none", edgecolors=PALETTE["red"],
               linewidths=1.2, zorder=6, label="Border ZIP centroid")
    # dispensaries
    ax.scatter(disp["lon"], disp["lat"], s=9, c=PALETTE["gold"], alpha=0.75, linewidths=0, zorder=7,
               label="Dispensary")

    labels = {"NM": (-106.1, 34.4), "CO": (-105.6, 38.55), "AZ": (-110.7, 33.6),
              "UT": (-111.3, 38.5), "TX": (-103.5, 32.1), "OK": (-103.0, 36.35)}
    for s, (x, y) in labels.items():
        tag = "legal" if s in LEGAL_NB else ("" if s == "NM" else "no legal market")
        ax.text(x, y, s + (f"\n({tag})" if tag else ""), fontsize=10, ha="center",
                fontweight="bold" if s == "NM" else "normal", color="#2B2B2B")

    ax.set_xlim(-112.2, -102.4)
    ax.set_ylim(31.0, 39.1)
    ax.set_axis_off()
    handles = [
        mpatches.Patch(color=GREEN, label="Neighbor with operating recreational market (CO, AZ)"),
        mpatches.Patch(color=GRAY, label="Neighbor with no recreational market (TX, OK, UT)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["gold"], alpha=0.8, markersize=7,
               label="Dispensary (New Mexico, 2022--2026)"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=PALETTE["red"], markersize=8,
               label="Border ZIP ($\\leq$75 mi of a nonlegal line, $\\geq$1,000 nonlegal residents)"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=PALETTE["blue"], markersize=7,
               label="Interior ZIP"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              fontsize=8.5, frameon=False)
    ax.set_title("The New Mexico setting: a free-entry legal market bordering nonlegal Texas, Oklahoma, and Utah",
                 fontsize=11.5, pad=12)
    fig.savefig(os.path.join(OUT, "fig_nm_setting.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_nm_setting.png")


# ----------------------------------------------------------------------
def fig_spatial_inequality():
    apply_healy_style()
    counties = load_counties()
    nm_diss = counties[counties["STUSPS"] == "NM"].dissolve(by="STUSPS")

    zb = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    zc = pd.read_csv(os.path.join(TBL, "nm_zip_panel_zeroinclusive.csv"), dtype={"zip5": str})
    dens = (zc.groupby("zip5")["dispensary_per_capita"].mean().rename("dens_per_1000").reset_index())
    z = zb.merge(dens, on="zip5", how="left")
    z["dens_per_1000"] = z["dens_per_1000"].fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1.15]})

    # --- (a) proportional-symbol map ---
    ax = axes[0]
    counties[counties["STUSPS"].isin(["AZ", "CO", "UT", "TX", "OK"])].plot(
        ax=ax, color=LIGHT, edgecolor="white", linewidth=0.3)
    counties[counties["STUSPS"] == "NM"].boundary.plot(ax=ax, edgecolor="#CFCFCF", linewidth=0.3)
    nm_diss.boundary.plot(ax=ax, edgecolor="#2B2B2B", linewidth=1.2)
    for grp, color, lab in [(0, PALETTE["blue"], "Interior ZIP"), (1, PALETTE["red"], "Border ZIP")]:
        s = z[z.border_zip == grp]
        ax.scatter(s["zip_lon"], s["zip_lat"], s=5 + 62 * np.sqrt(s["dens_per_1000"]),
                   c=color, alpha=0.5, edgecolors="white", linewidths=0.3, label=lab, zorder=4 + grp)
    for dv in [0.5, 1.5, 3.0]:
        ax.scatter([], [], s=5 + 62 * np.sqrt(dv), c="#999999", alpha=0.6, label=f"{dv:g} per 1,000")
    ax.set_xlim(-109.6, -102.6)
    ax.set_ylim(31.0, 37.3)
    ax.set_axis_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=5, fontsize=8, frameon=False,
              handletextpad=0.3, columnspacing=0.9)
    ax.set_title("(a) Dispensaries per 1,000 residents, by ZIP code", fontsize=11)

    # --- (b) border vs. interior, population-weighted ---
    ax = axes[1]
    zc2 = zc.merge(zb[["zip5", "border_zip"]], on="zip5", how="left", suffixes=("", "_b"))
    zc2["bz"] = zc2["border_zip"].fillna(zc2.get("border_zip_b"))
    grp = zc2.groupby("bz").agg(disp=("n_dispensaries", "sum"),
                                pop=("zip_population", "sum"),
                                sales=("total_sales", "sum"))
    grp["dens"] = 1000 * grp["disp"] / grp["pop"]
    grp["rpc"] = grp["sales"] / grp["pop"]
    order = [1.0, 0.0]
    names = {1.0: "Border ZIPs", 0.0: "Interior ZIPs"}
    cols = {1.0: PALETTE["red"], 0.0: PALETTE["blue"]}

    ax2 = ax.twinx()
    w = 0.32
    xs = np.arange(2)
    for k in range(2):
        g = order[k]
        ax.bar(xs[k] - w / 2, grp.loc[g, "dens"], width=w, color=cols[g], alpha=0.9, edgecolor="white")
        ax2.bar(xs[k] + w / 2, grp.loc[g, "rpc"], width=w, color=cols[g], alpha=0.45, edgecolor="white", hatch="//")
        ax.text(xs[k] - w / 2, grp.loc[g, "dens"], f"{grp.loc[g,'dens']:.2f}", ha="center", va="bottom", fontsize=9)
        ax2.text(xs[k] + w / 2, grp.loc[g, "rpc"], f"\\${grp.loc[g,'rpc']:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([names[order[0]], names[order[1]]], fontsize=10)
    ax.set_xlim(-0.65, 1.65)
    ax.set_ylabel("Dispensaries per 1,000 residents (solid)")
    ax2.set_ylabel("Revenue per resident, \\$/quarter (hatched)")
    ax.set_ylim(0, grp["dens"].max() * 1.25)
    ax2.set_ylim(0, grp["rpc"].max() * 1.25)
    ax2.grid(False)
    ratio_d = grp.loc[1.0, "dens"] / grp.loc[0.0, "dens"]
    ratio_r = grp.loc[1.0, "rpc"] / grp.loc[0.0, "rpc"]
    ax.set_title(f"(b) Border ZIPs support {ratio_d:.1f}x the dispensary density and\n"
                 f"{ratio_r:.1f}x the revenue per resident of interior ZIPs (population-weighted)",
                 fontsize=10.5)

    fig.suptitle("Spatial inequality in dispensary access, New Mexico (2022Q2--2026Q3 pooled)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_nm_spatial_inequality.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_nm_spatial_inequality.png")


# ----------------------------------------------------------------------
def fig1_southwest():
    apply_healy_style()
    counties = load_counties()

    ps = pd.read_csv(os.path.join(EG, "tables", "master_perstore_panel.csv"), dtype={"fips5": str})
    d25 = ps[ps.year == 2025].copy()
    d25["dens"] = 1000 * pd.to_numeric(d25["dispensary_count"], errors="coerce") / d25["population"]

    bc = adjacency_border_counties(("NM", "CO"))

    m = counties.merge(d25[["fips5", "dens", "state_abbr"]], left_on="GEOID", right_on="fips5", how="left")
    m["border_county"] = m["GEOID"].map(bc)

    sample = m[m["STUSPS"].isin(["CO", "NM"])].copy()
    neighbors = ["UT", "WY", "NE", "KS", "OK", "TX", "AZ"]

    fig, ax = plt.subplots(figsize=(11, 9))
    counties[counties["STUSPS"].isin(neighbors)].plot(ax=ax, color=GRAY, edgecolor="white", linewidth=0.3)
    # mark the one legal neighbor
    counties[counties["STUSPS"] == "AZ"].dissolve(by="STUSPS").plot(ax=ax, color=GREEN, edgecolor="white", linewidth=0.3, alpha=0.5)

    vmax = np.nanpercentile(sample["dens"], 97)
    sample.plot(ax=ax, column="dens", cmap="Blues", edgecolor="white", linewidth=0.25,
                vmin=0, vmax=vmax, missing_kwds={"color": LIGHT, "edgecolor": "white", "linewidth": 0.25})

    border = sample[sample["border_county"] == 1]
    border.boundary.plot(ax=ax, edgecolor=PALETTE["red"], linewidth=1.3, zorder=5)

    for st in ["CO", "NM"]:
        counties[counties["STUSPS"] == st].dissolve(by="STUSPS").boundary.plot(
            ax=ax, edgecolor="#2B2B2B", linewidth=1.3, zorder=6)

    for s, (x, y) in {"NM": (-106.1, 34.3), "CO": (-105.5, 38.7), "AZ": (-110.6, 34.5),
                      "UT": (-111.2, 39.3), "TX": (-102.2, 32.9), "OK": (-100.95, 35.6),
                      "KS": (-101.7, 38.6), "NE": (-103.4, 41.35), "WY": (-107.5, 41.35)}.items():
        ax.text(x, y, s, fontsize=9.5, ha="center", color="#4D4D4D",
                fontweight="bold" if s in ("NM", "CO") else "normal")

    ax.set_xlim(-111.7, -100.9)
    ax.set_ylim(31.0, 41.6)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.42, pad=0.01, aspect=22, location="right")
    cbar.set_label("Dispensaries per 1,000 residents (2025)", fontsize=9.5)

    n_co_b = int((border.STUSPS == "CO").sum())
    n_nm_b = int((border.STUSPS == "NM").sum())
    handles = [
        mpatches.Patch(facecolor="none", edgecolor="#2B2B2B", linewidth=1.3, label="State boundary (CO, NM)"),
        mpatches.Patch(facecolor="none", edgecolor=PALETTE["red"], linewidth=1.3,
                       label=f"Border county: shares a land border with a nonlegal state "
                             f"({n_nm_b} NM, {n_co_b} CO)"),
        mpatches.Patch(color=GRAY, label="Neighboring nonlegal state (TX, OK, UT, WY, NE, KS)"),
        mpatches.Patch(color=GREEN, alpha=0.5, label="Neighboring legal state (AZ)"),
        mpatches.Patch(color=LIGHT, label="County with no dispensary in the 2025 estimation sample"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1,
              fontsize=8.5, frameon=False)
    ax.set_title("Dispensary density and the nonlegal-state border: New Mexico and Colorado, 2025",
                 fontsize=12, pad=10)
    fig.savefig(os.path.join(OUT, "fig1_southwest_map.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig1_southwest_map.png  (border counties outlined: {n_nm_b} NM, {n_co_b} CO)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_setting()
    fig_spatial_inequality()
    fig1_southwest()
