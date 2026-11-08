"""
Dynamic border-premium event study (border vs. interior, by quarter) for the
New Mexico ZIP-quarter panel, estimated under TWO border-status definitions:

  1. PREDETERMINED / geography-only ("border_zip"): a ZIP is border if its own
     centroid is within 75 miles of the nonlegal state line (TX/OK/UT) AND at
     least 1,000 nonlegal-state residents live within 75 miles of that centroid.
     No dispensary or revenue data enters the classification.
     Source: tables/nm_zip_border_stateline.csv (already merged into
     nm_zip_panel_corrected.csv). This is the definition used in the paper.

  2. DISPENSARY-ANCHORED / outcome-dependent ("border_zip_disp"): a ZIP is
     border if it contains at least one dispensary whose OWN location satisfies
     the same 75-mile / 1,000-resident rule (border_dispensary == 1 in
     nm_dispensary_quarterly_panel_v3.csv, taken as max() over the ZIP's
     dispensaries). This is the pre-revision definition; treatment status
     depends on where dispensaries actually chose to locate.

The two definitions agree on 113 of the 116 ZIPs in the regression sample and
differ on 3 (88119, 88220, 88337), which are interior by centroid but contain a
border-classified dispensary -> 38 border ZIPs under (1), 41 under (2).

Specification (identical to analyze_nm_zip_border_premium.py's dynamic profile):

  log(y_zt) = alpha + beta * Border_z
              + sum_{q != q0} beta_q * (Border_z x Quarter_q)
              + gamma * log(Population_z) + delta_q + eps_zt

Reference quarter q0 = 2022Q2 (first quarter of the panel; the ZIP panel is
entirely post-legalization, so this traces whether the border-interior gap is
stable or evolves, it is not a pre/post legalization event study). Each quarter
coefficient is converted back to that quarter's own LEVEL via t_test
(beta + beta_q), not left as a deviation from the reference quarter. Standard
errors clustered by county throughout.

Outputs (New Mexico project/figures and tables/):
  fig_nm_zip_eventstudy_borderdef_comparison.png   -- 3 outcomes, both defs overlaid
  fig_nm_zip_eventstudy_predetermined.png          -- 3 outcomes, definition 1 only
  fig_nm_zip_eventstudy_dispensaryanchored.png     -- 3 outcomes, definition 2 only
  eventstudy_nm_zip_borderdef_<def>_<outcome>.csv  -- coefficient tables
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")

OUTCOMES = [
    ("log_revenue_per_capita", "Revenue per capita"),
    ("log_dispensary_per_capita", "Dispensary density"),
    ("log_revenue_per_dispensary", "Revenue per dispensary"),
]
REF_QUARTER = "2022Q2"
DEFS = {
    "predetermined": ("border_zip", "Predetermined (ZIP centroid, 75 mi)"),
    "dispensaryanchored": ("border_zip_disp", "Dispensary-anchored (contains a border dispensary)"),
}


def load_panel():
    df = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"),
                     dtype={"zip5": str, "county_fips": str})
    v3 = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"),
                     dtype={"zip5_nearest": str})
    disp = (v3.groupby("zip5_nearest")["border_dispensary"].max()
            .rename("border_zip_disp").reset_index()
            .rename(columns={"zip5_nearest": "zip5"}))
    df = df.merge(disp, on="zip5", how="left")
    df["border_zip_disp"] = df["border_zip_disp"].fillna(0).astype(int)
    return df


def run_dynamic(df, border_var, yvar):
    d = df.dropna(subset=[yvar, "log_population"]).copy()
    quarters = sorted(d["yq"].unique())
    formula = (f"{yvar} ~ {border_var}*C(yq, Treatment(reference='{REF_QUARTER}')) "
               f"+ log_population")
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["county_fips"]})
    rows = []
    for q in quarters:
        if q == REF_QUARTER:
            b, se, p = m.params[border_var], m.bse[border_var], m.pvalues[border_var]
        else:
            term = f"{border_var}:C(yq, Treatment(reference='{REF_QUARTER}'))[T.{q}]"
            if term not in m.params.index:
                continue
            tt = m.t_test(f"{border_var} + {term}")
            b = float(np.ravel(tt.effect)[0])
            se = float(np.ravel(tt.sd)[0])
            p = float(np.ravel(tt.pvalue)[0])
        rows.append({"quarter": q, "coef": b, "se": se, "p": p})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_panel()
    print(f"Sample: {len(df)} ZIP-quarter obs, {df.zip5.nunique()} ZIPs, "
          f"{df.county_fips.nunique()} counties, {df.yq.nunique()} quarters "
          f"({min(df.yq)}-{max(df.yq)})")
    zz = df.drop_duplicates("zip5")
    print(f"Border ZIPs -- predetermined: {int(zz.border_zip.sum())}, "
          f"dispensary-anchored: {int(zz.border_zip_disp.sum())}, "
          f"disagree on {int((zz.border_zip != zz.border_zip_disp).sum())} ZIPs\n")

    apply_healy_style()
    results = {k: {} for k in DEFS}
    for def_key, (bvar, def_label) in DEFS.items():
        print(f"=== {def_label} [{bvar}] ===")
        for yvar, label in OUTCOMES:
            ev = run_dynamic(df, bvar, yvar)
            results[def_key][label] = ev
            ev.to_csv(os.path.join(TBL, f"eventstudy_nm_zip_borderdef_{def_key}_{yvar}.csv"),
                      index=False)
            n_sig = int((ev["p"] < 0.05).sum())
            print(f"  {label:<24} {n_sig}/{len(ev)} quarters p<0.05, "
                  f"mean level {ev['coef'].mean():+.3f}, "
                  f"range [{ev['coef'].min():+.2f}, {ev['coef'].max():+.2f}]")
        print()

    # ---- individual figures (one per definition) ----
    for def_key, (bvar, def_label) in DEFS.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, (yvar, label) in zip(axes, OUTCOMES):
            ev = results[def_key][label]
            xt = np.arange(len(ev))
            ax.errorbar(xt, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=3,
                        markersize=4, linewidth=1.2, color=PALETTE["blue"])
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xticks(xt)
            ax.set_xticklabels(ev["quarter"], rotation=90, fontsize=7)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Quarter")
        axes[0].set_ylabel("Border coefficient (log points)")
        fig.suptitle(f"Dynamic border premium by quarter, New Mexico ZIP panel\n"
                     f"{def_label}; reference quarter {REF_QUARTER}; county-clustered 95% CI",
                     fontsize=12)
        fig.tight_layout()
        path = os.path.join(OUT, f"fig_nm_zip_eventstudy_{def_key}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")

    # ---- comparison figure (both definitions overlaid) ----
    colors = {"predetermined": PALETTE["blue"], "dispensaryanchored": PALETTE["red"]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (yvar, label) in zip(axes, OUTCOMES):
        for j, (def_key, (bvar, def_label)) in enumerate(DEFS.items()):
            ev = results[def_key][label]
            xt = np.arange(len(ev)) + (j - 0.5) * 0.16
            ax.errorbar(xt, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=2,
                        markersize=3.5, linewidth=1.1, color=colors[def_key],
                        label=def_label if ax is axes[0] else None, alpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(np.arange(len(results["predetermined"][label])))
        ax.set_xticklabels(results["predetermined"][label]["quarter"], rotation=90, fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Quarter")
    axes[0].set_ylabel("Border coefficient (log points)")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Dynamic border premium by quarter, New Mexico ZIP panel: "
                 "predetermined vs. dispensary-anchored border status\n"
                 f"reference quarter {REF_QUARTER}; county-clustered 95% CI; "
                 "quarter levels (border main effect + interaction)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_nm_zip_eventstudy_borderdef_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
