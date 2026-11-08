"""
Continuous dose-response analysis for New Mexico: does dispensary/ZIP
outcomes respond smoothly (or non-monotonically) to distance from the
nonlegal-state line, rather than the binary border/interior split used
elsewhere in this paper? Distance is binned into 25-mile bands, letting
the data reveal the shape of the relationship rather than imposing a
linear or polynomial functional form. The farthest band (175-200 miles)
is dropped entirely rather than used as the reference: 75 percent of its
observations are Bernalillo County (Albuquerque), whose ZIPs average 2.7
times the population of the nearest-border ZIPs, an urban-agglomeration
confound unrelated to border distance. The nearest-border band (0-25
miles) is the reference category instead.

Specification: log(y) ~ C(dist_bin) + log(Population) + C(period),
county-clustered SEs. Each bin's coefficient is a direct, correctly-
signed marginal effect relative to the 0-25-mile reference bin (no
t_test-based level extraction needed here, unlike the event-study
figures elsewhere in this project, because distance bins enter as a
single categorical variable rather than an interaction with a border
main effect).

Three outcomes, NM only:
  - Revenue per capita (dispensary level)
  - Revenue per dispensary (dispensary level, i.e. a dispensary's own
    revenue -- the "per store" analog at the individual-firm level)
  - Dispensary density (ZIP level, log_density_zip)
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

BIN_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-125", "125-150", "150-175"]
BIN_ORDER = BIN_LABELS  # 0-25mi (nearest border) is first -> reference category


def fmt_pct_plain(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def run_spec(df, yvar, popvar, cluster_var, label, tag):
    d = df.dropna(subset=[yvar, "dist_bin", popvar, cluster_var]).copy()
    d["dist_bin"] = d["dist_bin"].cat.remove_unused_categories()
    formula = f"{yvar} ~ C(dist_bin) + {popvar} + C(period)"
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster_var]})

    rows = []
    for b in BIN_ORDER:
        if b == BIN_ORDER[0]:
            rows.append((b, 0.0, 0.0, np.nan, len(d[d.dist_bin == b])))
            continue
        key = f"C(dist_bin)[T.{b}]"
        if key not in m.params.index:
            continue
        coef, se, p = m.params[key], m.bse[key], m.pvalues[key]
        rows.append((b, coef, se, p, len(d[d.dist_bin == b])))
    dd = pd.DataFrame(rows, columns=["dist_bin", "coef", "se", "p", "n"])
    dd.to_csv(os.path.join(OUT, f"doseresponse_nm_{tag}.csv"), index=False)
    print(f"[{label}] n={len(d)}, R2={m.rsquared:.3f}")
    print(dd.to_string(index=False))
    print()
    return dd, m


def main():
    os.makedirs(OUT, exist_ok=True)
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_doseresponse_panel.csv"), dtype={"fips5": str})
    disp["dist_bin"] = pd.Categorical(disp["dist_bin"], categories=BIN_ORDER, ordered=False)
    zipp = pd.read_csv(os.path.join(TBL, "nm_zip_doseresponse_panel.csv"), dtype={"fips5": str})
    zipp["dist_bin"] = pd.Categorical(zipp["dist_bin"], categories=BIN_ORDER, ordered=False)

    specs = [
        ("Revenue per capita", disp, "log_revenue_percapita", "log_population", "fips5", "revpercapita"),
        ("Revenue per dispensary", disp, "log_revenue", "log_population", "fips5", "revperdisp"),
        ("Dispensary density", zipp, "log_density_zip", "log_zip_population", "fips5", "density"),
    ]

    results = {}
    for label, df, yvar, popvar, cluster_var, tag in specs:
        dd, m = run_spec(df, yvar, popvar, cluster_var, label, tag)
        results[label] = dd

    # ---------- table ----------
    tex_path = os.path.join(OUT, "table_nm_doseresponse.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Dose-response of dispensary outcomes to distance from the nonlegal-state line, New Mexico}
\label{tab:nm_doseresponse}
\footnotesize
\begin{tabular}{lccc}
\toprule
Distance band (mi) & Revenue per capita & Revenue per dispensary & Dispensary density \\
\midrule
""")
        for b in BIN_ORDER:
            cells = [b]
            for label in ["Revenue per capita", "Revenue per dispensary", "Dispensary density"]:
                dd = results[label]
                row = dd[dd.dist_bin == b]
                if len(row) == 0:
                    cells.append("--")
                    continue
                coef, p = row.iloc[0]["coef"], row.iloc[0]["p"]
                if b == BIN_ORDER[0]:
                    cells.append("ref.")
                else:
                    cells.append(f"${coef:+.3f}$ (${fmt_pct(coef)}$), $p={p:.3f}$")
            f.write(" & ".join(cells) + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Each column is a separate regression of the named log outcome on a full set of 25-mile
distance-band indicators (omitted reference: 0-25 miles, immediately adjacent to the nonlegal-state line), a
log population control, and quarter fixed effects. The 175-200 mile band is excluded entirely: 75 percent of
its observations are Bernalillo County (Albuquerque), whose ZIPs average 2.7 times the population of the
nearest-border band, an urban-agglomeration confound unrelated to border distance. Revenue per capita and
revenue per dispensary are estimated at the individual dispensary level; dispensary density is estimated at
the ZIP level (log dispensaries per 1,000 ZIP residents). County-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"Wrote {tex_path}")

    # ---------- figure ----------
    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, label in zip(axes, ["Revenue per capita", "Revenue per dispensary", "Dispensary density"]):
        dd = results[label]
        xt = np.arange(len(dd))
        ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                    linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(dd["dist_bin"], rotation=45, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Distance from nonlegal-state line (miles)")
        ax.set_ylabel("Coefficient vs. 0-25mi reference (log points)")
    fig.suptitle("Dose-response: distance from the nonlegal-state line, New Mexico\n"
                 "25-mile distance bands, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_nm_doseresponse.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
