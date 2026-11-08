"""
Dynamic (event-study-style) border-vs-interior difference in revenue per
LOCAL resident, where "local resident" = 2020 Census population within 1 km
of the dispensary (nm_dispensary_local_population.csv, pop_1km_areal).

Outcome:  y_it = log( total_sales_it / pop_1km_areal_i )
Spec (dynamic form matching the project's ZIP dynamic-border-premium
convention -- fig_nm_dynamic_borderpremium.png -- which interacts Border with
EVERY quarter and omits the base Border term, so each quarter's coefficient is
the border-interior gap in that quarter directly, with no single omitted
reference period; controls follow analyze_eventstudy_quarterly_NM_dispensary_best.py):

    y_it = a + sum_q g_q (Border_i x 1[period=q])
             + phi*log(pop_1km_areal_i) + delta_q + brand_i + e_it

  - g_q : border-interior difference in log(revenue per 1 km resident) in quarter q
  - Border_i : predetermined 75 mi / 1,000-nonlegal-resident state-line rule
               on the dispensary's own geocoded location
  - county-clustered SEs (fips5) -- county clustering beat ZIP clustering in
    every grid-search cell for the revenue outcomes
  - brand FE: mechanically-extracted chain label; lowers revenue-outcome SEs
  - log(pop_1km) kept as a control even though it is the denominator, matching
    the ZIP-baseline / county-dispensary house convention (lets the local-
    population elasticity differ from -1)

A pooled static Border coefficient is also reported for reference.

Denominator hygiene: a 1 km radius leaves a low-population tail (37 dispensaries
< 100 residents, 82 < 250) that would otherwise dominate the variance (same
problem the project hit with local_density's original 10 mi radius). Primary
sample drops dispensaries with pop_1km_areal < 100 and winsorises the outcome
at its 1st/99th percentile. A 2 km-denominator version is run as robustness.
2026Q3 is dropped as a partial revenue quarter.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
sys.path.insert(0, os.path.join(BASE, "..", "Latest try", "code"))
try:
    from healy_style import apply_healy_style, PALETTE
    apply_healy_style()
    C1, C2 = PALETTE["blue"], PALETTE["red"]
except Exception:
    C1, C2 = "#2166ac", "#b2182b"

DROP_Q = {"2026Q3"}
POP_FLOOR = 100.0
REF_LABEL = "2022Q2"


def build(dic_radius_col, pop_floor):
    v3 = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"), dtype={"fips5": str})
    loc = pd.read_csv(os.path.join(TBL, "nm_dispensary_local_population.csv"))
    d = v3.merge(loc[["dispensary_id", dic_radius_col]], on="dispensary_id", how="left")
    d = d[~d["yq"].isin(DROP_Q)].copy()
    d = d[(d["total_sales"] > 0) & (d[dic_radius_col] >= pop_floor)].copy()
    d["y"] = np.log(d["total_sales"] / d[dic_radius_col])
    d["log_localpop"] = np.log(d[dic_radius_col])
    lo, hi = d["y"].quantile([0.01, 0.99])
    d["y"] = d["y"].clip(lo, hi)
    # contiguous period index on the retained quarters, ref = 2022Q2
    periods = sorted(d["yq"].unique(), key=lambda s: (int(s[:4]), int(s[-1])))
    pmap = {p: i for i, p in enumerate(periods)}
    d["per"] = d["yq"].map(pmap)
    return d, periods


def estimate(d, periods):
    # dynamic: Border interacted with every quarter, NO base Border term, so
    # each g_q is that quarter's border-interior gap. Build explicit columns
    # to keep coefficient names predictable.
    d = d.copy()
    pers = sorted(d["per"].unique())
    bcols = []
    for p in pers:
        col = f"b_q{p}"
        d[col] = ((d["per"] == p) & (d["border_dispensary"] == 1)).astype(float)
        bcols.append(col)
    m = smf.ols(
        "y ~ " + " + ".join(bcols) + " + log_localpop + C(per) + C(brand)",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    rows = []
    for p, col in zip(pers, bcols):
        rows.append((periods[p], m.params[col], m.bse[col], m.pvalues[col]))
    es = pd.DataFrame(rows, columns=["yq", "coef", "se", "p"])
    # static
    ms = smf.ols("y ~ border_dispensary + log_localpop + C(per) + C(brand)",
                 data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    static = (ms.params["border_dispensary"], ms.bse["border_dispensary"],
              ms.pvalues["border_dispensary"], int(ms.nobs), d["dispensary_id"].nunique())
    return es, static


def main():
    results = {}
    for tag, col in [("1km", "pop_1km_areal"), ("2km", "pop_2km_areal")]:
        floor = POP_FLOOR if tag == "1km" else 250.0
        d, periods = build(col, floor)
        es, static = estimate(d, periods)
        es.to_csv(os.path.join(OUT, f"eventstudy_nm_dispensary_revper{tag}resident.csv"), index=False)
        results[tag] = (es, static)
        b, se, p, n, nd = static
        print(f"[{col}] n={n} dispensary-quarters, {nd} dispensaries, {len(periods)} quarters "
              f"({periods[0]}..{periods[-1]})")
        print(f"    static border effect on log(revenue / {tag} residents): "
              f"b={b:+.3f} ({(np.exp(b)-1)*100:+.1f}%), se={se:.3f}, p={p:.3f}")

    # ---- main figure: 1 km ----
    es, static = results["1km"]
    x = np.arange(len(es))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="gray", lw=0.9, ls="--")
    ax.errorbar(x, es["coef"], yerr=1.96 * es["se"], fmt="o-", color=C1, capsize=3,
                markersize=5, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(es["yq"], rotation=90, fontsize=8)
    ax.set_ylabel("Border $-$ interior gap in log(revenue per 1 km resident)")
    b, se, p, n, nd = static
    ax.set_title(
        "Dynamic border$-$interior difference in dispensary revenue per local resident\n"
        "local resident = 2020 Census population within 1 km; border $\\times$ quarter, "
        "county-clustered 95% CI\n"
        f"static pooled effect: {(np.exp(b)-1)*100:+.1f}% (p={p:.3f}), "
        f"n={n:,} dispensary-quarters, {nd:,} dispensaries",
        fontsize=10.5)
    fig.tight_layout()
    f1 = os.path.join(OUT, "fig_eventstudy_nm_dispensary_revper1kmresident.png")
    fig.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nWrote", f1)

    # ---- robustness figure: 1 km vs 2 km ----
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="gray", lw=0.9, ls="--")
    for (tag, c) in [("1km", C1), ("2km", C2)]:
        e = results[tag][0]
        xx = np.arange(len(e))
        ax.errorbar(xx + (0.12 if tag == "2km" else -0.12), e["coef"], yerr=1.96 * e["se"],
                    fmt="o-", color=c, capsize=2, markersize=4, lw=1.2,
                    label=f"denominator = pop within {tag}")
    ax.set_xticks(np.arange(len(results["1km"][0])))
    ax.set_xticklabels(results["1km"][0]["yq"], rotation=90, fontsize=8)
    ax.set_ylabel("Border $-$ interior gap in log(revenue per local resident)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Robustness: revenue-per-local-resident border gap, 1 km vs 2 km denominator\n"
                 "border $\\times$ quarter; county-clustered 95% CI", fontsize=10.5)
    fig.tight_layout()
    f2 = os.path.join(OUT, "fig_eventstudy_nm_dispensary_revperresident_1km_vs_2km.png")
    fig.savefig(f2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", f2)


if __name__ == "__main__":
    main()
