"""
Robustness / placebo check on the NM ZIP-code border design, per explicit
user request: instead of comparing border ZIP codes (near a NONLEGAL
neighbor) against true interior, compare them against ZIP codes that are
ALSO near a state line but specifically near an already-LEGAL neighbor
(Colorado or Arizona). True interior ZIP codes (not within 50 miles of
either kind of neighbor) are dropped from this specific check.

  - Treatment = 1: ZIP code within 50 miles of the nearest OK or TX city
    (population >= 5,000) -- the nonlegal-neighbor side, same distance
    methodology as the main 50-mile ZIP design, restricted to OK/TX only
    (not "any nonlegal state nationwide", per the explicit instruction).
  - Control (treatment = 0): ZIP code within 50 miles of the nearest CO or
    AZ city (population >= 5,000) -- the legal-neighbor side.

If the earlier border premium reflects genuine cross-border demand from a
NONLEGAL market (not just some generic "being near any state line"
confound -- through-traffic, highway commerce, etc.), it should show up
here too: treatment (OK/TX-adjacent) should carry a premium relative to
control (CO/AZ-adjacent), even though both groups are equally "near a
state line."

Produces both a main pooled table (treatment effect, one row per outcome)
and a border x quarter event study, saved as .tex + .png / .png
respectively, clustered by county.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
THRESHOLD_MI = 50
POP_MIN = 5000
EARTH_RADIUS_MI = 3958.8

NONLEGAL_STATES = ["OK", "TX"]
LEGAL_NEIGHBOR_STATES = ["CO", "AZ"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def load_cities(states):
    gaz = pd.read_csv(os.path.join(SOURCE_BASE, "data", "2023_Gaz_place_national.txt"), sep="\t")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["state_fips"] = (gaz["GEOID"] // 100000).astype(int)
    gaz["place_fips"] = (gaz["GEOID"] % 100000).astype(int)
    pop = pd.read_csv(os.path.join(SOURCE_BASE, "data", "sub-est2023.csv"), encoding="latin1")
    pop = pop[pop["SUMLEV"] == 162][["STATE", "PLACE", "POPESTIMATE2023"]].copy()
    pop = pop.rename(columns={"STATE": "state_fips", "PLACE": "place_fips"})
    cities = gaz.merge(pop, on=["state_fips", "place_fips"], how="inner")
    cities = cities[cities["USPS"].isin(states) & (cities["POPESTIMATE2023"] >= POP_MIN)].copy()
    return cities


def nearest_dist(zips, cities):
    dm = haversine_miles(zips["zip_lat"].to_numpy()[:, None], zips["zip_lon"].to_numpy()[:, None],
                          cities["INTPTLAT"].to_numpy()[None, :], cities["INTPTLONG"].to_numpy()[None, :])
    idx = dm.argmin(axis=1)
    return dm[np.arange(len(zips)), idx], cities["NAME"].to_numpy()[idx], cities["USPS"].to_numpy()[idx]


def build_panel():
    zref = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
    zref = zref.rename(columns={"zip": "zip5", "lat": "zip_lat", "lng": "zip_lon", "population": "zip_population"})
    zref = zref[["zip5", "zip_lat", "zip_lon", "zip_population", "county_fips", "county_name"]]

    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_panel_monthly_zip.csv"), dtype={"zip5": str})
    zips = disp[["zip5"]].drop_duplicates().merge(zref, on="zip5", how="left")
    zips = zips.dropna(subset=["zip_lat"]).copy()

    okTX = load_cities(NONLEGAL_STATES)
    coAZ = load_cities(LEGAL_NEIGHBOR_STATES)
    print(f"OK/TX city universe: {len(okTX)} cities. CO/AZ city universe: {len(coAZ)} cities.")

    zips["dist_okTX_mi"], zips["nearest_okTX_city"], zips["nearest_okTX_state"] = nearest_dist(zips, okTX)
    zips["dist_coAZ_mi"], zips["nearest_coAZ_city"], zips["nearest_coAZ_state"] = nearest_dist(zips, coAZ)

    zips["treatment_zip"] = (zips["dist_okTX_mi"] <= THRESHOLD_MI).astype(int)
    zips["control_zip"] = ((zips["dist_coAZ_mi"] <= THRESHOLD_MI) & (zips["treatment_zip"] == 0)).astype(int)
    overlap = ((zips["dist_okTX_mi"] <= THRESHOLD_MI) & (zips["dist_coAZ_mi"] <= THRESHOLD_MI)).sum()
    print(f"ZIP codes within {THRESHOLD_MI}mi of BOTH an OK/TX and a CO/AZ city (should be ~0 given NM's width): "
          f"{overlap}")

    print(f"\nTreatment (OK/TX-adjacent, <= {THRESHOLD_MI}mi): {zips.treatment_zip.sum()} ZIP codes")
    print(zips[zips.treatment_zip == 1][["zip5", "county_name", "dist_okTX_mi", "nearest_okTX_city"]]
          .sort_values("dist_okTX_mi").to_string(index=False))
    print(f"\nControl (CO/AZ-adjacent, <= {THRESHOLD_MI}mi): {zips.control_zip.sum()} ZIP codes")
    print(zips[zips.control_zip == 1][["zip5", "county_name", "dist_coAZ_mi", "nearest_coAZ_city"]]
          .sort_values("dist_coAZ_mi").to_string(index=False))

    zips.to_csv(os.path.join(TBL, "nm_zip_placebo_classification.csv"), index=False)

    disp["quarter"] = ((disp["month"] - 1) // 3) + 1
    q = disp.groupby(["zip5", "year", "quarter"]).agg(
        total_sales=("total_sales", "sum"), dispensary_count=("dispensary_id", "nunique"),
        n_months=("month", "nunique")).reset_index()
    q = q[q["n_months"] == 3].drop(columns="n_months")
    q = q.merge(zips[["zip5", "zip_population", "county_fips", "county_name", "treatment_zip", "control_zip"]],
                on="zip5", how="left")
    q = q[q["zip_population"].notna() & (q["zip_population"] > 0)].copy()
    q = q[(q.treatment_zip == 1) | (q.control_zip == 1)].copy()

    q["revenue_per_capita"] = q["total_sales"] / q["zip_population"]
    q["revenue_per_store"] = q["total_sales"] / q["dispensary_count"]
    q["dispensary_per_1000"] = 1000 * q["dispensary_count"] / q["zip_population"]
    for col in ["revenue_per_capita", "revenue_per_store", "dispensary_per_1000"]:
        q[f"log_{col}"] = np.log(q[col].where(q[col] > 0))
    q["log_population"] = np.log(q["zip_population"])
    q["state_abbr"] = "NM"
    q["yq"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)
    q["period"] = q["yq"]

    out = os.path.join(TBL, "nm_zip_quarterly_panel_placebo.csv")
    q.to_csv(out, index=False)
    print(f"\nWrote {out}: {q.shape}, {q.zip5.nunique()} ZIP codes "
          f"({q[q.treatment_zip==1].zip5.nunique()} treatment, {q[q.control_zip==1].zip5.nunique()} control), "
          f"periods {q.yq.min()}-{q.yq.max()}")
    return q


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def main_table(q):
    specs = [("Revenue per capita", "log_revenue_per_capita"),
             ("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]
    rows_tex, rows_png = [], []
    for label, yvar in specs:
        d = q.dropna(subset=[yvar, "treatment_zip", "log_population"]).copy()
        formula = f"{yvar} ~ treatment_zip + log_population + C(period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
        b = m.params.get("treatment_zip", np.nan)
        p = m.pvalues.get("treatment_zip", np.nan)
        n = int(m.nobs)
        n_treat = d[d.treatment_zip == 1].zip5.nunique()
        n_ctrl = d[d.treatment_zip == 0].zip5.nunique()
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & {n} & {n_treat} & {n_ctrl} \\\\\n")
        rows_png.append([label, fmt_pct(b), f"{p:.3f}", n, n_treat, n_ctrl])
        print(f"{label}: n={n} (treat ZIPs={n_treat}, control ZIPs={n_ctrl}), "
              f"treatment effect={fmt_pct(b)} (b={b:+.3f}, p={p:.3f})")

    tex_path = os.path.join(OUT, "table_nm_zip_placebo_legalneighbor.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Placebo check: OK/TX-adjacent (treatment) vs.\ CO/AZ-adjacent (control) ZIP codes, New Mexico}
\label{tab:nm_zip_placebo}
\footnotesize
\begin{tabular}{lccccc}
\toprule
Outcome & Coef. (\%) & $p$ (cl.) & $n$ & Treatment ZIPs & Control ZIPs \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.95\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Sample restricted to ZIP codes within 50 miles of either an OK/TX city (treatment, population
$\geq$5,000) or a CO/AZ city (control), dropping true interior ZIP codes near neither. County-clustered
standard errors.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")

    fig, ax = plt.subplots(figsize=(9, 0.6 * (len(rows_png) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows_png, colLabels=["Outcome", "Treatment effect", "p (cl.)", "n", "Treat ZIPs", "Ctrl ZIPs"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title("NM placebo check: OK/TX-adjacent vs. CO/AZ-adjacent ZIP codes", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_nm_zip_placebo_legalneighbor.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def event_study(q):
    specs = [("Revenue per capita", "log_revenue_per_capita"),
             ("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]
    es_results = {}
    for label, yvar in specs:
        d = q.dropna(subset=[yvar, "treatment_zip", "log_population"]).copy()
        formula = f"{yvar} ~ treatment_zip + treatment_zip:C(period) + log_population + C(period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})

        all_periods = sorted(d["period"].unique())
        interacted = {t.split("[T.")[1].rstrip("]") for t in m.params.index
                      if t.startswith("treatment_zip:C(period)")}
        rows = []
        for p in all_periods:
            test_str = (f"treatment_zip + treatment_zip:C(period)[T.{p}] = 0" if p in interacted
                        else "treatment_zip = 0")
            tt = m.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        es_results[label] = dd
        dd.to_csv(os.path.join(OUT, f"eventstudy_nm_zip_placebo_{label.replace(' ', '')}.csv"), index=False)
        print(f"{label}: n={len(d)}, {len(all_periods)} quarters, "
              f"treatment main (reference period) effect b={m.params.get('treatment_zip', float('nan')):+.3f} "
              f"p={m.pvalues.get('treatment_zip', float('nan')):.3f}")

    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (label, dd) in zip(axes, es_results.items()):
        xt = np.arange(len(dd))
        ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                    linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(dd["period"], rotation=90, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Treatment coefficient (log points)")
    fig.suptitle("Placebo event study: OK/TX-adjacent (treatment) vs. CO/AZ-adjacent (control) ZIP codes, NM\n"
                  "county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_nm_zip_placebo.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    q = build_panel()
    main_table(q)
    event_study(q)
