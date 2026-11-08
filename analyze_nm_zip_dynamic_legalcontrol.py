"""
Robustness check: the New Mexico dynamic border-premium specification with a
different control group.

The main design (Figure~\ref{fig:dynamic_borderpremium}) compares border ZIP
codes -- within 75 miles of the Texas/Oklahoma/Utah line, with at least 1,000
nonlegal-state residents in reach -- against all other (interior) New Mexico ZIP
codes. Here we replace the interior control group with ZIP codes that are also
near a state line, but a line shared with a state that already has a legal
recreational market (Colorado or Arizona). The "legal-border" control ZIP codes
are defined with the *same rule* as the treated border ZIP codes: centroid
within 75 miles of the Colorado/Arizona line, with at least 1,000 Colorado or
Arizona residents within 75 miles. True interior ZIP codes, near neither kind of
line, are dropped.

If the border premium reflects genuine demand from consumers in nonlegal states
rather than a generic feature of being near any state line, it should survive
this comparison: treated ZIP codes should still carry the premium relative to
ZIP codes that are equally close to a border but face a neighbor whose residents
can already buy cannabis at home.

Specification is identical to analyze_nm_zip_border_premium.py's dynamic profile:
  log(y_zt) = alpha + beta*Border_z + sum_{q != q0} beta_q (Border_z x Quarter_q)
              + gamma*log(Population_z) + delta_q + eps_zt
reference quarter q0 = 2022Q2; quarter coefficients reported as levels
(beta + beta_q via t_test); county-clustered standard errors.

Outputs:
  figures and tables/fig_nm_dynamic_legalcontrol.png
  tables/nm_zip_legalborder_stateline.csv
  tables/nm_zip_dynamic_legalcontrol_<outcome>.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
EG = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
SHP = os.path.join(EG, "data", "us_counties_shp", "cb_2023_us_county_500k.shp")

REF_QUARTER = "2022Q2"
LEGAL_NEIGHBORS = ["CO", "AZ"]          # New Mexico's legal-market neighbors
OUTCOMES = [
    ("log_revenue_per_capita", "Revenue per capita"),
    ("log_dispensary_per_capita", "Dispensary density"),
    ("log_revenue_per_dispensary", "Revenue per dispensary"),
]


def haversine_mi(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def build_legalborder():
    """Legal-neighbor analogue of build_nm_zip_border_stateline.py."""
    zips = pd.read_csv(os.path.join(EG, "data", "raw", "uszips_simplemaps.csv"),
                       dtype={"zip": str, "county_fips": str})
    nm = zips[zips.state_id == "NM"].dropna(subset=["lat", "lng", "population"]).copy()
    nm = nm[nm.population > 0]

    states = gpd.read_file(SHP)[["STUSPS", "geometry"]].rename(columns={"STUSPS": "state_abbr"}).to_crs(5070)
    states = states.dissolve(by="state_abbr").reset_index()
    legal_union = states[states.state_abbr.isin(LEGAL_NEIGHBORS)].union_all()

    pts = gpd.GeoSeries(gpd.points_from_xy(nm["lng"], nm["lat"]), crs=4326).to_crs(5070)
    nm["dist_legalline_mi"] = pts.distance(legal_union).to_numpy() / 1609.34

    ln = zips[zips["state_id"].isin(LEGAL_NEIGHBORS)].dropna(subset=["lat", "lng", "population"])
    ln_lat, ln_lon, ln_pop = ln["lat"].to_numpy(), ln["lng"].to_numpy(), ln["population"].to_numpy()
    legal_pop = np.array([ln_pop[haversine_mi(la, lo, ln_lat, ln_lon) <= 75].sum()
                          for la, lo in zip(nm["lat"].to_numpy(), nm["lng"].to_numpy())])
    nm["legal_pop_within_75mi"] = legal_pop
    nm["legal_border_zip"] = ((nm["dist_legalline_mi"] <= 75) & (legal_pop >= 1000)).astype(int)

    out = nm.rename(columns={"zip": "zip5"})[
        ["zip5", "dist_legalline_mi", "legal_pop_within_75mi", "legal_border_zip"]]
    out.to_csv(os.path.join(TBL, "nm_zip_legalborder_stateline.csv"), index=False)
    print(f"Legal-border ZIP codes (75mi of CO/AZ line, >=1,000 CO/AZ residents in reach): "
          f"{int(out.legal_border_zip.sum())} of {len(out)} populated NM ZIPs")
    return out


def run_dynamic(df, yvar):
    d = df.dropna(subset=[yvar, "log_population"]).copy()
    quarters = sorted(d["yq"].unique())
    formula = f"{yvar} ~ border_zip*C(yq, Treatment(reference='{REF_QUARTER}')) + log_population"
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
    lb = build_legalborder()

    df = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"),
                     dtype={"zip5": str, "county_fips": str})
    df = df.merge(lb[["zip5", "legal_border_zip"]], on="zip5", how="left")
    df["legal_border_zip"] = df["legal_border_zip"].fillna(0).astype(int)

    # treated = nonlegal-facing border ZIPs; control = legal-facing border ZIPs
    # (a ZIP that is near BOTH a nonlegal and a legal line stays with the treated group)
    keep = (df.border_zip == 1) | ((df.legal_border_zip == 1) & (df.border_zip == 0))
    sample = df[keep].copy()
    zz = sample.drop_duplicates("zip5")
    n_treat = int((zz.border_zip == 1).sum())
    n_ctrl = int(((zz.border_zip == 0) & (zz.legal_border_zip == 1)).sum())
    print(f"\nSample: {sample.zip5.nunique()} ZIP codes "
          f"({n_treat} border / {n_ctrl} legal-border control), "
          f"{len(sample)} ZIP-quarters, {sample.county_fips.nunique()} counties")

    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (yvar, label) in zip(axes, OUTCOMES):
        ev = run_dynamic(sample, yvar)
        ev.to_csv(os.path.join(TBL, f"nm_zip_dynamic_legalcontrol_{yvar}.csv"), index=False)
        n_sig = int((ev["p"] < 0.05).sum())
        print(f"  {label:<22} {n_sig}/{len(ev)} quarters p<0.05, "
              f"mean level {ev['coef'].mean():+.3f}, range [{ev['coef'].min():+.2f}, {ev['coef'].max():+.2f}]")
        xt = np.arange(len(ev))
        ax.errorbar(xt, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=3, markersize=4,
                    linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(ev["quarter"], rotation=90, fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Quarter")
    axes[0].set_ylabel("Border coefficient (log points)")
    fig.suptitle("Dynamic border premium with a legal-border control group, New Mexico\n"
                 f"treated = border ZIP codes (near TX/OK/UT); control = ZIP codes near the CO/AZ line, "
                 f"same 75-mile rule.\nReference quarter {REF_QUARTER}; county-clustered 95% CI",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_nm_dynamic_legalcontrol.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
