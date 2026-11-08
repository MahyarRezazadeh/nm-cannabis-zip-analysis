"""
Same specification as analyze_county_border_capped_main.py, using the
100-mile border definition (county centroid within 100mi of the nonlegal
state line, >=1,000 nonlegal-state residents within that reach) instead of
the original 75mi cutoff -- see build_county_border_stateline_100mi.py.
Border-county count goes from 56 (75mi) to 88 (100mi) across CO/OR/MT/NM.
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


def fmt_pct_plain(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def load():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    border = pd.read_csv(os.path.join(TBL, "county_border_stateline_100mi.csv"), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county", "dist_stateline_mi"]], on="fips5", how="left")
    df["is_capped"] = df["entry_capped"]
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    print(f"{len(df)} county-quarter rows, {df.fips5.nunique()} counties, "
          f"{df.drop_duplicates('fips5').border_county.sum()} border counties")
    return df


def main_table(df):
    specs = ["log_revenue_per_capita", "log_revenue_per_store", "log_dispensary_per_1000"]
    labels = ["Revenue per capita", "Revenue per store", "Dispensary density"]
    rows_tex, rows_png = [], []
    for label, yvar in zip(labels, specs):
        d = df.dropna(subset=[yvar, "border_county", "is_capped"]).copy()
        formula = f"{yvar} ~ border_county + border_county:is_capped + C(state_period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b_b, p_b = m.params.get("border_county", np.nan), m.pvalues.get("border_county", np.nan)
        b_i, p_i = m.params.get("border_county:is_capped", np.nan), m.pvalues.get("border_county:is_capped", np.nan)
        n = int(m.nobs)
        n_border = d.drop_duplicates("fips5").border_county.sum()
        rows_tex.append(f"{label} & ${b_b:+.3f}$ (${fmt_pct(b_b)}$) & {p_b:.3f} & "
                         f"${b_i:+.3f}$ (${fmt_pct(b_i)}$) & {p_i:.3f} & {n} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b_b), f"{p_b:.3f}", fmt_pct_plain(b_i), f"{p_i:.3f}", n])
        print(f"{label}: n={n}, border counties={n_border}, border={fmt_pct_plain(b_b)} (p={p_b:.3f}), "
              f"border x capped={fmt_pct_plain(b_i)} (p={p_i:.3f})")

    tex_path = os.path.join(OUT, "table_county_border_capped_main_100mi.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{County-level border effect and entry-cap interaction, 100-mile border definition (CO, OR, MT, NM; quarterly)}
\label{tab:county_border_capped_main_100mi}
\footnotesize
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{Border} & \multicolumn{2}{c}{Border $\times$ Capped} & \\
Outcome & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) & $n$ \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Sample: CO, OR, MT, NM (Washington dropped). Border = county centroid within 100 miles of the
nonlegal-state line, with $\geq$1,000 nonlegal-state residents within that reach (loosened from the 75-mile main
definition in Table~\ref{tab:county_border_capped_main}; 88 of 189 counties qualify as border here vs.\ 56 under
75 miles). Capped is the state-year entry-cap indicator (CO/MT always capped, NM always free, OR switches capped
in 2022). State and state $\times$ quarter fixed effects via a single concatenated state-quarter key. County-clustered
standard errors.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")

    fig, ax = plt.subplots(figsize=(11, 0.6 * (len(rows_png) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows_png, colLabels=["Outcome", "Border", "p (cl.)", "Border x Capped", "p (cl.)", "n"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    ax.set_title("County-level border effect x entry-cap, 100mi definition (CO, OR, MT, NM)", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_county_border_capped_main_100mi.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_county_border_capped_main_100mi.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    main_table(df)
