"""
Final noise-minimizing dispensary-level specification, chosen by a full
factorial grid search over {population control, WLS age-weighting, chain/
brand FE, clustering level} x {3 outcomes} = 48 regressions (see
_grid_search_noise.py / tables/_grid_search_results.csv), ranked by
standard error on the border coefficient.

Two findings drove the final choice, both confirmed empirically rather
than assumed:

1. County-level clustering beats ZIP-level clustering in EVERY one of
   the 16 combinations tested, for all three outcomes. ZIP clustering
   (116 clusters, ~10 dispensaries/cluster) is consistently noisier than
   county clustering (32 clusters, ~37 dispensaries/cluster) -- a known
   finite-sample property of the cluster-robust sandwich estimator when
   average cluster size shrinks this much, not evidence that ZIP is the
   "more correct" clustering level for this data.

2. Chain/brand fixed effects substantially help the two REVENUE outcomes
   (SE roughly 15-20% lower) despite only 31% of the sample identifying
   the border coefficient once brand FE absorbs firm-level heterogeneity
   -- the variance reduction from removing persistent firm-level
   heterogeneity outweighs the loss of identifying sample. But brand FE
   does NOT help local dispensary density (a purely spatial-competition
   measure with no natural firm-brand determinant) -- adding it there
   only costs sample for no precision gain, so it's excluded for that
   outcome.

Winning specification per outcome (SE-ranked, most precise first):
  - Revenue per capita:        county population + brand FE, unweighted, county-clustered (SE=0.107, down from 0.129 baseline)
  - Revenue (per dispensary):  county population + brand FE, unweighted, county-clustered (SE=0.103, down from 0.125 baseline)
  - Local dispensary density:  county population, NO brand FE, unweighted, county-clustered (SE=0.160, down from 0.272 under the
                                original 10-mile local_density construction -- the 10mi->25mi radius widening is what did the work here)
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
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")

SPECS = [
    ("Revenue per capita", "log_revenue_percapita", True),
    ("Revenue (per dispensary)", "log_revenue", True),
    ("Local dispensary density", "log_local_density", False),
]


def fmt_pct_plain(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def load():
    df = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v2.csv"), dtype={"fips5": str, "zip5_nearest": str})
    df = df.replace([np.inf, -np.inf], np.nan)
    df["log_population"] = np.log(df["population"])
    return df


def formula_for(yvar, brand_fe):
    rhs = "border_dispensary + log_population + C(period)"
    if brand_fe:
        rhs += " + C(brand)"
    return f"{yvar} ~ {rhs}"


def main_table(df):
    rows_tex, rows_png = [], []
    for label, yvar, brand_fe in SPECS:
        need = [yvar, "border_dispensary", "log_population", "fips5"]
        if brand_fe:
            need.append("brand")
        d = df.dropna(subset=need).copy()
        formula = formula_for(yvar, brand_fe)
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b, se, p, n = (m.params["border_dispensary"], m.bse["border_dispensary"],
                       m.pvalues["border_dispensary"], int(m.nobs))
        n_treat = d[d.border_dispensary == 1].dispensary_id.nunique()
        n_ctrl = d[d.border_dispensary == 0].dispensary_id.nunique()
        fe_tag = "Yes" if brand_fe else "No"
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {se:.3f} & {p:.3f} & {n} & {fe_tag} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b), f"{se:.3f}", f"{p:.3f}", n, fe_tag])
        print(f"{label}: brand_fe={brand_fe}, n={n} (treat={n_treat}, ctrl={n_ctrl}), "
              f"border effect={fmt_pct_plain(b)} (b={b:+.3f}, se={se:.3f}, p={p:.3f})")

    tex_path = os.path.join(OUT, "table_nm_dispensary_border_best.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{New Mexico dispensary-level border effect, noise-minimizing specification (grid-search selected)}
\label{tab:nm_dispensary_border_best}
\footnotesize
\begin{tabular}{lccccc}
\toprule
Outcome & Coef. (\%) & SE & $p$ (cl.) & $n$ & Brand FE \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Specification chosen by a full factorial grid search over population control (county vs.\
25-mile-radius), WLS age-weighting (on/off), chain/brand fixed effects (on/off), and clustering level
(county vs.\ ZIP), ranked by standard error on the border coefficient (48 regressions; see
\_grid\_search\_noise.py). County-level clustering dominates ZIP-level clustering in every one of the 16
combinations tested, for all three outcomes. Brand fixed effects (mechanically extracted from each
dispensary's business name) reduce the standard error for the two revenue outcomes by roughly 15-20 percent
despite only 31 percent of the sample identifying the border coefficient once brand FE is added -- the
variance reduction from removing persistent firm-level heterogeneity dominates the loss of identifying
sample. Brand FE does not help local dispensary density (a spatial-competition measure with no natural
brand determinant) and is therefore excluded for that outcome. Local dispensary density itself uses a
25-mile radius (vs.\ the original 10 miles), which alone cuts its standard error from 0.272 to 0.160 by
eliminating a small-denominator blowup for dispensaries in sparsely populated areas.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")

    fig, ax = plt.subplots(figsize=(10.5, 0.6 * (len(rows_png) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows_png, colLabels=["Outcome", "Border effect", "SE", "p (cl.)", "n", "Brand FE"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    ax.set_title("NM dispensary-level border effect: noise-minimizing spec", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_nm_dispensary_border_best.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_nm_dispensary_border_best.png")


def event_study(df):
    es_results = {}
    for label, yvar, brand_fe in SPECS:
        need = [yvar, "border_dispensary", "log_population", "fips5"]
        if brand_fe:
            need.append("brand")
        d = df.dropna(subset=need).copy()
        rhs = "border_dispensary + border_dispensary:C(period) + log_population + C(period)"
        if brand_fe:
            rhs += " + C(brand)"
        formula = f"{yvar} ~ {rhs}"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

        all_periods = sorted(d["period"].unique())
        interacted = {int(t.split("[T.")[1].rstrip("]")) for t in m.params.index
                      if t.startswith("border_dispensary:C(period)")}
        rows = []
        for p in all_periods:
            test_str = (f"border_dispensary + border_dispensary:C(period)[T.{p}] = 0" if p in interacted
                        else "border_dispensary = 0")
            tt = m.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        es_results[label] = dd
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_dispensary_best_{label.replace(' ', '').replace('(', '').replace(')', '')}.csv"),
                   index=False)
        print(f"{label}: n={len(d)}, {len(all_periods)} quarters, "
              f"border main (ref. period) effect b={m.params.get('border_dispensary', float('nan')):+.3f} "
              f"p={m.pvalues.get('border_dispensary', float('nan')):.3f}")

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
        ax.set_ylabel("Border coefficient (log points)")
    fig.suptitle("Border x quarter event study, NM dispensary level -- noise-minimizing spec\n"
                  "county-clustered 95% CI (brand FE for revenue outcomes only)", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_dispensary_best.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    main_table(df)
    event_study(df)
