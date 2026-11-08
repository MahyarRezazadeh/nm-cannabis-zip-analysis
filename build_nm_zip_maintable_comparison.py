"""
Main (non-event-study, single border coefficient per outcome) table for the
NM ZIP-code design, comparing all three specifications built this session:
  (1) Original: border_zip + log_population + C(period) + C(state_abbr)
  (2) + population-within-70mi catchment control
  (3) State/quarter/state x quarter FE only (no catchment control) --
      numerically identical to (1) for this NM-only panel, kept as its own
      column since it was run and requested separately.
Clustered by county throughout.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fit(d, formula, yvar):
    dd = d.dropna(subset=[c for c in [yvar, "border_zip", "log_population", "log_population_within_70mi"]
                           if c in d.columns]).copy()
    m = smf.ols(formula, data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["county_fips"]})
    return m.params.get("border_zip", np.nan), m.pvalues.get("border_zip", np.nan), int(m.nobs)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(TBL, "nm_zip_quarterly_panel.csv"), dtype={"zip5": str, "county_fips": str})
    df["period"] = df["yq"]
    catch = pd.read_csv(os.path.join(TBL, "nm_zip_catchment_population.csv"), dtype={"zip5": str})
    df = df.merge(catch[["zip5", "population_within_70mi"]], on="zip5", how="left")
    df["log_population_within_70mi"] = np.log(df["population_within_70mi"])

    specs_outcome = [("Revenue per capita", "log_revenue_per_capita"),
                      ("Revenue per store", "log_revenue_per_store"),
                      ("Dispensary density", "log_dispensary_per_1000")]

    formulas = {
        "Original": "{y} ~ border_zip + log_population + C(period) + C(state_abbr)",
        "+ catchment pop.": "{y} ~ border_zip + log_population + log_population_within_70mi + "
                             "C(period) + C(state_abbr) + C(state_abbr):C(period)",
        "State/qtr/state x qtr FE": "{y} ~ border_zip + log_population + C(period) + C(state_abbr) "
                                     "+ C(state_abbr):C(period)",
    }

    rows_tex, rows_png = [], []
    results = {}
    for label, yvar in specs_outcome:
        row_tex = [label]
        row_png = [label]
        for spec_name, tmpl in formulas.items():
            b, p, n = fit(df, tmpl.format(y=yvar), yvar)
            results[(label, spec_name)] = (b, p, n)
            row_tex.append(f"${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f}")
            row_png.append(f"{fmt_pct(b)}\n(p={p:.3f})")
            print(f"{label} | {spec_name}: b={b:+.3f} ({fmt_pct(b)}), p={p:.3f}, n={n}")
        rows_tex.append(" & ".join(row_tex) + r" \\" + "\n")
        rows_png.append(row_png)

    tex_path = os.path.join(OUT, "table_nm_zip_maintable_comparison.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{New Mexico ZIP-code border effect, main specification comparison}
\label{tab:nm_zip_maintable_comparison}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Original} & \multicolumn{2}{c}{+ catchment pop.} & \multicolumn{2}{c}{State/qtr/state$\times$qtr FE} \\
Outcome & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Border = ZIP code within 50 miles of the nearest incorporated place (population $\geq$5,000)
in a state with no operating recreational retail market (OK/TX for New Mexico); interior otherwise, including
ZIP codes near the already-legal AZ/CO borders. ``Original'' and ``State/qtr/state$\times$qtr FE'' are
numerically identical for this NM-only panel: state and state$\times$quarter fixed effects add no identifying
variation beyond quarter fixed effects when only one state is present. ``+ catchment pop.'' additionally
controls for log(population within 70 miles), computed using all US ZIP codes (not just New Mexico's), which
is systematically larger for border ZIP codes (El Paso, TX inflates it) and pulls part of the raw border
coefficient toward zero once controlled for. County-clustered standard errors throughout.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")

    fig, ax = plt.subplots(figsize=(11, 0.7 * (len(rows_png) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows_png, colLabels=["Outcome", "Original", "+ catchment pop.", "State/qtr/stateXqtr FE"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    ax.set_title("NM ZIP-code border effect: specification comparison", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_nm_zip_maintable_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote table_nm_zip_maintable_comparison.png")


if __name__ == "__main__":
    main()
