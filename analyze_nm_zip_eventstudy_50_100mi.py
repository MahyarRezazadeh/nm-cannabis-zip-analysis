"""
Appendix event-study figures: the dynamic border-premium profile of the paper's
Figure 2 (fig_nm_dynamic_borderpremium.png), re-estimated under the 50-mile and
100-mile ZIP-level border definitions instead of the 75-mile baseline.

Border definition (identical rule to build_nm_zip_border_stateline.py, only the
threshold changes): a ZIP is border if its centroid lies within D miles of the
nonlegal state line (TX/OK/UT) AND at least 1,000 nonlegal-state residents live
within D miles of that centroid, for D in {50, 100}. Purely geographic, fixed.

Specification (identical to analyze_nm_zip_border_premium.py's dynamic profile):

  log(y_zt) = alpha + beta*Border_z
              + sum_{q != q0} beta_q*(Border_z x Quarter_q)
              + gamma*log(Population_z) + delta_q + eps_zt

reference quarter q0 = 2022Q2; each quarter coefficient reported as that
quarter's own LEVEL (beta + beta_q via t_test); county-clustered SEs.

Output: one figure, three panels (one per outcome), each panel overlaying the
50-mile and 100-mile event-study series.
  figures and tables/fig_nm_zip_eventstudy_50_100mi.png
  tables/eventstudy_nm_zip_<50mi|100mi>_<outcome>.csv
  tables/nm_zip_border_stateline_<50|100>mi.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
EARTH_RADIUS_MI = 3958.8
REF_QUARTER = "2022Q2"
THRESHOLDS = [50, 100]
OUTCOMES = [
    ("log_revenue_per_capita", "Revenue per capita"),
    ("log_dispensary_per_capita", "Dispensary density"),
    ("log_revenue_per_dispensary", "Revenue per dispensary"),
]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def build_border(threshold_mi):
    """Reproduce build_nm_zip_border_stateline.py with a configurable threshold."""
    zips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"),
                       dtype={"zip": str, "county_fips": str})
    nm = zips[zips.state_id == "NM"].dropna(subset=["lat", "lng", "population"]).copy()
    nm = nm[nm.population > 0]

    legal = pd.read_csv(os.path.join(SOURCE_BASE, "data", "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2023-12-31")
    legal_states = set(legal.loc[legal["retail_date"].notna() & (legal["retail_date"] <= cutoff),
                                 "state_abbr"])

    shp = gpd.read_file(os.path.join(SOURCE_BASE, "data", "us_counties_shp",
                                    "cb_2023_us_county_500k.shp"))[["STUSPS", "geometry"]]
    shp = shp.rename(columns={"STUSPS": "state_abbr"}).to_crs(5070)
    states_poly = shp.dissolve(by="state_abbr").reset_index()
    nonlegal_states = states_poly[~states_poly.state_abbr.isin(legal_states)
                                  & (states_poly.state_abbr != "NM")]
    nonlegal_union = nonlegal_states.union_all()

    pts = gpd.GeoSeries([Point(xy) for xy in zip(nm["lng"], nm["lat"])], crs=4326).to_crs(5070)
    nm["dist_stateline_mi"] = pts.distance(nonlegal_union).to_numpy() / 1609.34

    allzips = zips.dropna(subset=["lat", "lng", "population"])
    nz = allzips[~allzips["state_id"].isin(legal_states)]
    nz_lat, nz_lon, nz_pop = nz["lat"].to_numpy(), nz["lng"].to_numpy(), nz["population"].to_numpy()

    pop_within = np.zeros(len(nm))
    for i, (lat, lon) in enumerate(zip(nm["lat"].to_numpy(), nm["lng"].to_numpy())):
        d = haversine_miles(lat, lon, nz_lat, nz_lon)
        pop_within[i] = nz_pop[d <= threshold_mi].sum()
    nm[f"nonlegal_pop_within_{threshold_mi}mi"] = pop_within
    nm["border_zip"] = ((nm["dist_stateline_mi"] <= threshold_mi) & (pop_within >= 1000)).astype(int)

    out = nm.rename(columns={"zip": "zip5", "population": "zip_population"})[
        ["zip5", "zip_population", "county_fips", "county_name", "dist_stateline_mi",
         f"nonlegal_pop_within_{threshold_mi}mi", "border_zip"]]
    out.to_csv(os.path.join(TBL, f"nm_zip_border_stateline_{threshold_mi}mi.csv"), index=False)
    return out


def build_panel(border):
    """Reproduce build_nm_zip_panel_corrected.py's ZIP-quarter aggregation."""
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"),
                       dtype={"zip5_nearest": str})
    agg = disp.groupby(["zip5_nearest", "yq", "year", "quarter"], as_index=False).agg(
        total_sales=("total_sales", "sum"),
        n_dispensaries=("dispensary_id", "nunique"),
    ).rename(columns={"zip5_nearest": "zip5"})
    p = agg.merge(border[["zip5", "county_fips", "zip_population", "border_zip"]],
                  on="zip5", how="left").dropna(subset=["border_zip"]).copy()
    p["border_zip"] = p["border_zip"].astype(int)
    p["log_revenue_per_capita"] = np.log(p["total_sales"] / p["zip_population"])
    p["log_dispensary_per_capita"] = np.log(p["n_dispensaries"] / p["zip_population"] * 1000)
    p["log_revenue_per_dispensary"] = np.log(p["total_sales"] / p["n_dispensaries"])
    p["log_population"] = np.log(p["zip_population"])
    return p


def run_dynamic(df, yvar):
    d = df.dropna(subset=[yvar, "log_population"]).copy()
    quarters = sorted(d["yq"].unique())
    formula = (f"{yvar} ~ border_zip*C(yq, Treatment(reference='{REF_QUARTER}')) + log_population")
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
    rows = []
    for q in quarters:
        if q == REF_QUARTER:
            b, se, p = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
        else:
            term = f"border_zip:C(yq, Treatment(reference='{REF_QUARTER}'))[T.{q}]"
            if term not in m.params.index:
                continue
            tt = m.t_test(f"border_zip + {term}")
            b = float(np.ravel(tt.effect)[0]); se = float(np.ravel(tt.sd)[0])
            p = float(np.ravel(tt.pvalue)[0])
        rows.append({"quarter": q, "coef": b, "se": se, "p": p})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for D in THRESHOLDS:
        border = build_border(D)
        nb = int(border.border_zip.sum())
        panel = build_panel(border)
        zb = panel.loc[panel.border_zip == 1, "zip5"].nunique()
        zi = panel.loc[panel.border_zip == 0, "zip5"].nunique()
        print(f"\n=== {D}-mile definition: {nb} border ZIPs statewide; "
              f"regression sample {zb} border / {zi} interior ZIPs, "
              f"{len(panel)} ZIP-quarter obs ===")
        results[D] = {}
        for yvar, label in OUTCOMES:
            ev = run_dynamic(panel, yvar)
            results[D][label] = ev
            ev.to_csv(os.path.join(TBL, f"eventstudy_nm_zip_{D}mi_{yvar}.csv"), index=False)
            n_sig = int((ev["p"] < 0.05).sum())
            print(f"  {label:<24} {n_sig}/{len(ev)} quarters p<0.05, "
                  f"mean level {ev['coef'].mean():+.3f}")

    apply_healy_style()
    colors = {50: PALETTE["blue"], 100: PALETTE["red"]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (yvar, label) in zip(axes, OUTCOMES):
        for j, D in enumerate(THRESHOLDS):
            ev = results[D][label]
            xt = np.arange(len(ev)) + (j - 0.5) * 0.18
            ax.errorbar(xt, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=2,
                        markersize=3.5, linewidth=1.1, color=colors[D], alpha=0.9,
                        label=(f"{D}-mile border" if ax is axes[0] else None))
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ref = results[THRESHOLDS[0]][label]
        ax.set_xticks(np.arange(len(ref)))
        ax.set_xticklabels(ref["quarter"], rotation=90, fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Quarter")
    axes[0].set_ylabel("Border coefficient (log points)")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Dynamic border premium by quarter, New Mexico ZIP panel: "
                 "50-mile vs. 100-mile border definition\n"
                 f"reference quarter {REF_QUARTER}; quarter levels (border main effect + "
                 "interaction); county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_nm_zip_eventstudy_50_100mi.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
