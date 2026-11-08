"""
Cleanest possible free-vs-capped border premium comparison: NM (always
free) vs. CO+MT (always capped), excluding Oregon entirely -- OR
switches regime mid-panel and contributes to both sides, diluting a
clean two-regime comparison. Same specification and border definition as
the main pooled table (Table 7/9), just restricted to the two always-
one-regime state groups.

log(y_cst) = alpha + beta*Border_c + Border_c*Capped_c + delta_st + eps_cst
County-clustered SEs. Capped is constant within NM (0) and within CO/MT
(1) for the whole sample, so it's collinear with state FE alone (not
state x quarter) -- the state x quarter interaction is still needed
because CO's and MT's within-state quarter-to-quarter shocks are not
otherwise controlled for.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def fmt_pct_plain(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def run(border_file, tag, threshold_label):
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[df.state_abbr.isin(["CO", "MT", "NM"])].copy()
    border = pd.read_csv(os.path.join(TBL, border_file), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county"]], on="fips5", how="left")
    df["is_capped"] = df["entry_capped"]
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)

    print(f"[{tag}] {len(df)} county-quarter rows, {df.fips5.nunique()} counties "
          f"({df[df.state_abbr=='NM'].fips5.nunique()} NM, "
          f"{df[df.state_abbr.isin(['CO','MT'])].fips5.nunique()} CO+MT), "
          f"{df.drop_duplicates('fips5').border_county.sum()} border counties")

    specs = [("Revenue per capita", "log_revenue_per_capita"),
             ("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]
    rows_tex, rows_png = [], []
    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_county", "is_capped"]).copy()
        formula = f"{yvar} ~ border_county + border_county:is_capped + C(state_period)"
        y, X = dmatrices(formula, data=d, return_type="dataframe")
        rank = np.linalg.matrix_rank(X.values)
        if rank < X.shape[1]:
            print(f"[{tag}] {label}: RANK DEFICIENT ({X.shape[1]} cols, rank {rank}) -- skipped")
            continue
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b_b, p_b = m.params.get("border_county", np.nan), m.pvalues.get("border_county", np.nan)
        b_i, p_i = m.params.get("border_county:is_capped", np.nan), m.pvalues.get("border_county:is_capped", np.nan)
        n = int(m.nobs)
        rows_tex.append(f"{label} & ${b_b:+.3f}$ (${fmt_pct(b_b)}$) & {p_b:.3f} & "
                         f"${b_i:+.3f}$ (${fmt_pct(b_i)}$) & {p_i:.3f} & {n} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b_b), f"{p_b:.3f}", fmt_pct_plain(b_i), f"{p_i:.3f}", n])
        print(f"[{tag}] {label}: n={n}, border(free/NM)={fmt_pct_plain(b_b)} (p={p_b:.3f}), "
              f"border x capped(CO+MT vs NM)={fmt_pct_plain(b_i)} (p={p_i:.3f})")

    tex_path = os.path.join(OUT, f"table_county_border_capped_nmcomt_{tag}.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Border premium, free entry (New Mexico) vs.\ capped entry (Colorado + Montana), """ +
                threshold_label + r"""}
\label{tab:county_border_capped_nmcomt_""" + tag + r"""}
\footnotesize
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{Border (NM, free)} & \multicolumn{2}{c}{Border $\times$ Capped (CO+MT vs.\ NM)} & \\
Outcome & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) & $n$ \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Sample restricted to New Mexico (always free entry) and Colorado + Montana (always capped
entry) -- Oregon excluded entirely, since it switches regime mid-panel and would contribute to both sides of
this comparison. Border = county centroid within """ + threshold_label + r""" of the nonlegal-state line, with
$\geq$1,000 nonlegal-state residents within that reach. State and state $\times$ quarter fixed effects via a
single concatenated state-quarter key. County-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"Wrote {tex_path}\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    run("county_border_stateline.csv", "75mi", "75 miles")
    run("county_border_stateline_100mi.csv", "100mi", "100 miles")
