"""
Dispensary-level dynamic/event-study specification, ZIP-clustered, requested
as an extension of the ZIP-aggregated baseline: does an individual
dispensary's own revenue (per resident of its ZIP, and in level) respond to
its ZIP's border status, and does that response evolve over time?

Border status is the CURRENT, predetermined, geography-only ZIP classification
(tables/nm_zip_border_stateline.csv, merged via the dispensary's own
zip5_nearest) -- NOT the old border_dispensary column already present in
nm_dispensary_quarterly_panel_v3.csv, which is a dispensary-own-coordinate
version of the now-superseded rule (see implementation_notes.md).

Two outcomes:
  - log_revenue_per_resident = log(dispensary's own quarterly sales / its
    ZIP's population) -- NOT the same as the panel's existing
    log_revenue_percapita column, which divides by COUNTY population.
  - log_revenue = log(dispensary's own quarterly sales) -- i.e. "revenue per
    dispensary" taken literally at the one-dispensary-per-observation level.

Two model variants, because ZIP fixed effects and a time-invariant regressor
(border_zip, or log ZIP population) cannot both be estimated -- the FE
absorbs them exactly, same collinearity problem already documented for
county/state fixed effects elsewhere in this project (see
analyze_cross_state_regime_3state.py's own docstring):

  Model A (quarter FE + log ZIP population + ZIP-clustered SEs, no ZIP FE):
    satisfies the literal request (quarter FE, ZIP population included) and
    is directly comparable in spirit to the paper's existing ZIP-aggregated
    baseline/dynamic design, just at the dispensary level with ZIP (not
    county) clustering.
  Model B (ZIP fixed effects + quarter FE + ZIP-clustered SEs): a genuinely
    stronger design that absorbs every time-invariant ZIP-level confound
    (border status AND population both fall out of the regression as a
    result -- only the DYNAMIC border x quarter interaction survives, since
    that varies within ZIP over time even though the main effect doesn't).
    This is the specification that most directly answers "why didn't we
    already have this" -- it's estimable, but only in dynamic/event-study
    form, never as a static baseline.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
REF_PERIOD = "2022Q2"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def build():
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"), dtype={"zip5_nearest": str})
    border = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    n0 = len(disp)
    disp = disp.merge(
        border[["zip5", "border_zip"]].rename(columns={"zip5": "zip5_nearest"}),
        on="zip5_nearest", how="left")
    n_missing = disp["border_zip"].isna().sum()
    if n_missing:
        print(f"Dropping {n_missing} of {n0} dispensary-quarter rows: ZIP not in the predetermined "
              f"border-classification file")
        print(disp.loc[disp["border_zip"].isna(), "zip5_nearest"].unique())
    disp = disp.dropna(subset=["border_zip", "zip_population"]).copy()
    disp["border_zip"] = disp["border_zip"].astype(int)

    n_zero_sales = (disp["total_sales"] <= 0).sum()
    if n_zero_sales:
        print(f"Dropping {n_zero_sales} dispensary-quarter rows with total_sales <= 0 "
              f"(undefined under log, same convention as the rest of this project)")
    disp = disp[disp["total_sales"] > 0].copy()

    disp["revenue_per_resident"] = disp["total_sales"] / disp["zip_population"]
    disp["log_revenue_per_resident"] = np.log(disp["revenue_per_resident"])
    disp["log_zip_population"] = np.log(disp["zip_population"])
    # log_revenue already present (= log(total_sales)), kept as-is: "revenue per dispensary"
    # at the one-dispensary-per-row level.

    print(f"\nFinal dispensary-quarter panel: {len(disp)} rows, {disp.dispensary_id.nunique()} dispensaries, "
          f"{disp.zip5_nearest.nunique()} ZIPs, {disp.period.nunique() if 'period' in disp else disp.yq.nunique()} quarters")
    n_border_zips = disp.loc[disp.border_zip == 1, "zip5_nearest"].nunique()
    n_interior_zips = disp.loc[disp.border_zip == 0, "zip5_nearest"].nunique()
    print(f"Border ZIPs represented: {n_border_zips}, Interior ZIPs represented: {n_interior_zips}")
    return disp


OUTCOMES = [("log_revenue_per_resident", "Revenue per resident"),
            ("log_revenue", "Revenue per dispensary (own revenue)")]


def model_a_static_and_dynamic(df):
    """Quarter FE + log ZIP population, ZIP-clustered SEs. No ZIP FE."""
    print("\n" + "=" * 70)
    print("MODEL A: quarter FE + log(ZIP population), ZIP-clustered SEs (no ZIP FE)")
    print("=" * 70)
    rows_static = []
    dyn_results = {}
    for yvar, label in OUTCOMES:
        d = df.dropna(subset=[yvar, "log_zip_population", "border_zip"]).copy()

        # static baseline
        m0 = smf.ols(f"{yvar} ~ border_zip + log_zip_population + C(yq)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["zip5_nearest"]})
        b0, se0, p0 = m0.params["border_zip"], m0.bse["border_zip"], m0.pvalues["border_zip"]
        n0 = int(m0.nobs)
        n_zips = d.zip5_nearest.nunique()
        n_disp = d.dispensary_id.nunique()
        rows_static.append((label, b0, se0, p0, n0, n_zips, n_disp))
        print(f"\n[STATIC] {label}: b={b0:+.3f} ({se0:.3f}), {fmt_pct(b0)}, p={p0:.3f}, "
              f"n={n0} ({n_disp} dispensaries, {n_zips} ZIPs)")

        # dynamic (border x quarter)
        m1 = smf.ols(f"{yvar} ~ border_zip + border_zip:C(yq) + log_zip_population + C(yq)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["zip5_nearest"]})
        periods = sorted(d["yq"].unique())
        interacted = {t.split("[T.")[1].rstrip("]") for t in m1.params.index if t.startswith("border_zip:C(yq)")}
        rows = []
        for p in periods:
            test_str = (f"border_zip + border_zip:C(yq)[T.{p}] = 0" if p in interacted else "border_zip = 0")
            tt = m1.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        dyn_results[label] = dd
        print(f"[DYNAMIC] {label}: {len(periods)} quarters estimated, "
              f"reference-quarter level b={m1.params.get('border_zip', np.nan):+.3f}")

    static_df = pd.DataFrame(rows_static, columns=["outcome", "coef", "se", "p", "n", "n_zips", "n_dispensaries"])
    static_df.to_csv(os.path.join(TBL, "nm_dispensary_zipclustered_static_modelA.csv"), index=False)
    for label, dd in dyn_results.items():
        tag = label.replace(" ", "").replace("(", "").replace(")", "")
        dd.to_csv(os.path.join(TBL, f"nm_dispensary_zipclustered_dynamic_modelA_{tag}.csv"), index=False)
    return static_df, dyn_results


def model_b_zipFE_dynamic(df):
    """ZIP fixed effects + quarter FE, ZIP-clustered SEs. Border main effect
    and log population are absorbed by ZIP FE (both time-invariant); only the
    dynamic border x quarter interaction is estimable."""
    print("\n" + "=" * 70)
    print("MODEL B: ZIP fixed effects + quarter FE, ZIP-clustered SEs")
    print("(border_zip main effect and log ZIP population are absorbed by ZIP FE -- ")
    print(" not a bug, a direct consequence of both being time-invariant within ZIP)")
    print("=" * 70)
    dyn_results = {}
    for yvar, label in OUTCOMES:
        d = df.dropna(subset=[yvar, "border_zip"]).copy()
        d = d[np.isfinite(d[yvar])].copy()
        periods = sorted(d["yq"].unique())
        ref = REF_PERIOD if REF_PERIOD in periods else periods[0]
        dummy_cols = []
        for p in periods:
            if p == ref:
                continue
            col = f"bxq_{p}"
            d[col] = ((d["yq"] == p) & (d["border_zip"] == 1)).astype(float)
            dummy_cols.append(col)

        # PanelOLS entity/time index must be unique per row; since several
        # dispensaries share a ZIP-quarter, use dispensary as the panel entity
        # (each dispensary never switches ZIP, so a dispensary-level entity
        # nested within ZIP is not the same as a ZIP entity) and absorb ZIP
        # fixed effects explicitly via "other_effects" instead of entity_effects.
        d["_t"] = pd.Categorical(d["yq"]).codes
        d["_entity"] = d["dispensary_id"].astype(str)
        panel2 = d.set_index(["_entity", "_t"])
        y2 = panel2[yvar]
        X2 = panel2[dummy_cols]
        panel2["zip_cat"] = pd.Categorical(d["zip5_nearest"]).codes
        clusters_df = pd.DataFrame({"zip_cluster": d["zip5_nearest"].values}, index=panel2.index)
        mod = PanelOLS(y2, X2, time_effects=True, other_effects=panel2[["zip_cat"]],
                        drop_absorbed=True)
        m = mod.fit(cov_type="clustered", clusters=clusters_df)

        rows = []
        for p in periods:
            if p == ref:
                rows.append((p, 0.0, 0.0))
                continue
            key = f"bxq_{p}"
            if key not in m.params.index:
                rows.append((p, np.nan, np.nan))
                continue
            rows.append((p, m.params[key], m.std_errors[key]))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        dyn_results[label] = dd
        print(f"\n[ZIP-FE DYNAMIC] {label}: n={int(m.nobs)}, R2={m.rsquared:.3f}, "
              f"{d.zip5_nearest.nunique()} ZIPs, {d.dispensary_id.nunique()} dispensaries")
        print(dd.to_string(index=False))

    for label, dd in dyn_results.items():
        tag = label.replace(" ", "").replace("(", "").replace(")", "")
        dd.to_csv(os.path.join(TBL, f"nm_dispensary_zipFE_dynamic_modelB_{tag}.csv"), index=False)
    return dyn_results


def make_figure(dyn_a, dyn_b):
    apply_healy_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    for col, (_, label) in enumerate(OUTCOMES):
        ddA = dyn_a[label]
        ax = axes[0, col]
        xt = np.arange(len(ddA))
        ax.errorbar(xt, ddA["coef"], yerr=1.96 * ddA["se"], fmt="o-", capsize=3, markersize=4,
                    linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(ddA["period"], rotation=90, fontsize=8)
        ax.set_title(f"Model A (quarter FE + log pop): {label}", fontsize=10)
        ax.set_ylabel("Border coefficient (log points)")

        ddB = dyn_b[label]
        ax = axes[1, col]
        xt = np.arange(len(ddB))
        ax.errorbar(xt, ddB["coef"], yerr=1.96 * ddB["se"], fmt="o-", capsize=3, markersize=4,
                    linewidth=1.2, color=PALETTE["red"] if "red" in PALETTE else PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(ddB["period"], rotation=90, fontsize=8)
        ax.set_title(f"Model B (ZIP FE + quarter FE): {label}", fontsize=10)
        ax.set_ylabel("Border x quarter coefficient (log points)")
    fig.suptitle("Dispensary-level dynamic border premium, New Mexico\nZIP-clustered 95% CI", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_nm_dispensary_zip_dynamic_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    df = build()
    static_a, dyn_a = model_a_static_and_dynamic(df)
    dyn_b = model_b_zipFE_dynamic(df)
    make_figure(dyn_a, dyn_b)
