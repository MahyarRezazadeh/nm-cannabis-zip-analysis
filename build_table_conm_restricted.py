"""
LaTeX table for the CO+NM-restricted county-level revenue-per-store /
dispensary-density regression (the two states with genuinely time-varying
dispensary counts, vs. MT/OR's frozen snapshot counts). Same specification
as the 4-state main county table: border_county + border_county:is_capped +
state x quarter FE (concatenated key), county-clustered SEs.
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
    """LaTeX-escaped version -- an unescaped % starts a comment in LaTeX."""
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def main():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    border = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county"]], on="fips5", how="left")
    df["is_capped"] = df["entry_capped"]
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    conm = df[df.state_abbr.isin(["CO", "NM"])].copy()

    specs = [("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]
    rows_tex, rows_png = [], []
    for label, yvar in specs:
        d = conm.dropna(subset=[yvar, "border_county", "is_capped"]).copy()
        formula = f"{yvar} ~ border_county + border_county:is_capped + C(state_period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b, p = m.params["border_county"], m.pvalues["border_county"]
        inter, p_i = m.params["border_county:is_capped"], m.pvalues["border_county:is_capped"]
        n = int(m.nobs)
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & "
                         f"${inter:+.3f}$ (${fmt_pct(inter)}$) & {p_i:.3f} & {n} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b), f"{p:.3f}", fmt_pct_plain(inter), f"{p_i:.3f}", n])
        print(f"{label}: n={n}, border={fmt_pct_plain(b)} (p={p:.3f}), "
              f"border x capped={fmt_pct_plain(inter)} (p={p_i:.3f})")

    tex_path = os.path.join(OUT, "table_county_border_capped_CONM.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{County-level border effect, restricted to Colorado and New Mexico}
\label{tab:county_border_capped_conm}
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
\begin{minipage}{0.95\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Same specification as Table~\ref{tab:county_border_capped_main}, restricted to Colorado and
New Mexico -- the only two states with a genuinely time-varying (not single-snapshot) dispensary count at
this frequency. Montana and Oregon's dispensary counts are a fixed current-year snapshot repeated across every
period, which adds pure noise (not signal) to a quarterly panel; excluding them substantially tightens
dispensary density's estimate (from $p=0.281$ pooled across all 4 states to $p=0.027$ here). County-clustered
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
    ax.set_title("County-level border effect, CO+NM only", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_county_border_capped_CONM.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_county_border_capped_CONM.png")


if __name__ == "__main__":
    main()
