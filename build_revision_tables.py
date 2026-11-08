"""
Builds the four new LaTeX tables required by the empirical-strategy
revision (Table 1: NM baseline; Table 2: state-specific border premiums;
Table 3: cross-regime interaction across samples; Table 4: counterfactual
revenue benchmark), from the CSVs already written by:
  analyze_nm_zip_border_premium.py
  analyze_cross_state_regime_3state.py
  build_counterfactual_revenue_benchmark.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def stars(p):
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""


def table1():
    d = pd.read_csv(os.path.join(TBL, "nm_zip_border_baseline_results.csv"))
    rows = []
    for _, r in d.iterrows():
        rows.append(f"{r['outcome']} & ${r['coef']:+.3f}{stars(r['p'])}$ & ({r['se']:.3f}) & "
                     f"{fmt_pct(r['coef'])} & {r['p']:.3f} & {int(r['n'])} & {int(r['n_zips'])} \\\\\n")
    tex = r"""\begin{table}[htbp]
\centering
\caption{New Mexico baseline border-premium estimates, ZIP-quarter panel}
\label{tab:nm_baseline}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Outcome & Coef. & (SE) & Implied \% & $p$ & $N$ & ZIPs \\
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} $\log(y_{zt}) = \alpha + \beta\,\text{Border}_z + \gamma\log(\text{Population}_z) + \delta_t + \varepsilon_{zt}$.
Border$_z$ is a predetermined, ZIP-geography-only indicator: 1 if the ZIP's own centroid lies within 75 miles of the
nonlegal-state line (Texas, Oklahoma, or Utah) and at least 1,000 nonlegal-state residents live within that same
75-mile reach; dispensary locations play no role in the classification. Quarter fixed effects, county-clustered
standard errors. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
"""
    with open(os.path.join(OUT, "table_nm_baseline.tex"), "w") as f:
        f.write(tex)
    print("Wrote table_nm_baseline.tex")


def table2():
    d = pd.read_csv(os.path.join(TBL, "cross_state_border_premium_by_state.csv"))
    outcomes = d["outcome"].unique()
    rows = []
    for outcome in outcomes:
        cells = [outcome]
        for state in ["NM", "CO", "MT"]:
            r = d[(d.state == state) & (d.outcome == outcome)].iloc[0]
            cells.append(f"${r['coef']:+.3f}{stars(r['p'])}$")
        rows.append(" & ".join(cells) + r" \\" + "\n")
        cells_se = [""]
        for state in ["NM", "CO", "MT"]:
            r = d[(d.state == state) & (d.outcome == outcome)].iloc[0]
            cells_se.append(f"({r['se']:.3f})")
        rows.append(" & ".join(cells_se) + r" \\" + "\n")
    n_row = ["Observations"]
    counties_row = ["Border counties"]
    for state in ["NM", "CO", "MT"]:
        sub = d[(d.state == state) & (d.outcome == outcomes[0])].iloc[0]
        n_row.append(str(int(sub["n"])))
        counties_row.append(f"{int(sub['n_border_counties'])} of {int(sub['n_counties'])}")

    tex = r"""\begin{table}[htbp]
\centering
\caption{Border premium estimated separately by state}
\label{tab:state_specific}
\footnotesize
\begin{tabular}{lccc}
\toprule
Outcome & New Mexico & Colorado & Montana \\
\midrule
""" + "".join(rows) + r"\midrule" + "\n" + \
        " & ".join(n_row) + r" \\" + "\n" + \
        " & ".join(counties_row) + r" \\" + "\n" + \
        r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} $\log(y_{ct}) = \alpha + \beta_s\,\text{Border}_c + \delta_t + \varepsilon_{ct}$, estimated
separately within each state (equivalently, the state-specific slice of a specification with state $\times$
quarter fixed effects). Border$_c$ = 1 if county $c$'s centroid lies within 75 miles of the nonlegal-state
line, with $\geq$1,000 nonlegal-state residents within that reach. Sample truncated to 2025Q2 or earlier.
County-clustered standard errors. Colorado's border premium is not statistically distinguishable from zero on
revenue per capita or dispensary density; Montana's is closer in magnitude to New Mexico's. This heterogeneity
is discussed directly in the text rather than only in the pooled estimate. $^{*}p<0.10$, $^{**}p<0.05$,
$^{***}p<0.01$.
\end{minipage}
\end{table}
"""
    with open(os.path.join(OUT, "table_state_specific.tex"), "w") as f:
        f.write(tex)
    print("Wrote table_state_specific.tex")


def table3():
    d = pd.read_csv(os.path.join(TBL, "cross_state_regime_interaction_all_samples.csv"))
    samples = ["full_NM_CO_MT", "NM_CO", "NM_MT"]
    sample_labels = {"full_NM_CO_MT": "NM+CO+MT", "NM_CO": "NM+CO", "NM_MT": "NM+MT"}
    # need to re-map: 'sample' col in CSV uses label_states strings like "NM+CO+MT","NM+CO","NM+MT" already
    outcomes = d["outcome"].unique()
    rows = []
    for outcome in outcomes:
        cells = [outcome]
        for lab in ["NM+CO+MT", "NM+CO", "NM+MT"]:
            r = d[(d["sample"] == lab) & (d.outcome == outcome)].iloc[0]
            cells.append(f"${r['interaction_coef']:+.3f}{stars(r['interaction_p'])}$")
        rows.append(" & ".join(cells) + r" \\" + "\n")
        cells_se = [""]
        for lab in ["NM+CO+MT", "NM+CO", "NM+MT"]:
            r = d[(d["sample"] == lab) & (d.outcome == outcome)].iloc[0]
            cells_se.append(f"({r['interaction_se']:.3f})")
        rows.append(" & ".join(cells_se) + r" \\" + "\n")

    tex = r"""\begin{table}[htbp]
\centering
\caption{Border $\times$ Restricted interaction across samples}
\label{tab:regime_interaction}
\footnotesize
\begin{tabular}{lccc}
\toprule
Outcome & NM+CO+MT & NM+CO & NM+MT \\
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} $\log(y_{cst}) = \alpha + \beta\,\text{Border}_c + \theta\,(\text{Border}_c \times
\text{Restricted}_s) + \delta_{st} + \varepsilon_{cst}$. Restricted$_s$ = 1 for Colorado and Montana, 0 for
New Mexico; both Border$_c$ and Restricted$_s$ are time-invariant in this sample, so county fixed effects
cannot be added (they would exactly absorb the interaction). State $\times$ quarter fixed effects via a
concatenated state-quarter key; county-clustered standard errors. Because entry regime varies only across
three states, county clustering addresses within-county serial correlation but cannot rule out other
persistent state-level differences correlated with the regulatory regime -- we interpret $\theta$ as a
difference in the estimated border premium across regimes, not an unqualified causal treatment effect.
$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
"""
    with open(os.path.join(OUT, "table_regime_interaction.tex"), "w") as f:
        f.write(tex)
    print("Wrote table_regime_interaction.tex")


def table4():
    d = pd.read_csv(os.path.join(TBL, "counterfactual_revenue_benchmark.csv")).iloc[0]
    tex = r"""\begin{table}[htbp]
\centering
\caption{Model-implied dispensary-revenue benchmark, Colorado and Montana border counties, 2024}
\label{tab:counterfactual}
\footnotesize
\begin{tabular}{lc}
\toprule
Quantity & Value \\
\midrule
""" + \
        f"Estimated interaction coefficient $\\theta$ & ${d['theta']:+.3f}$ \\\\\n" + \
        f"Free-entry benchmark multiplier $e^{{-\\theta}}$ & {d['multiplier']:.3f} \\\\\n" + \
        r"\midrule" + "\n" + \
        f"Actual 2024 revenue, CO+MT border counties & \\${d['actual_revenue_2024']:,.0f} \\\\\n" + \
        f"Free-entry-benchmark counterfactual revenue & \\${d['counterfactual_revenue_2024']:,.0f} \\\\\n" + \
        f"Model-implied additional dispensary revenue & \\${d['revenue_gap']:,.0f} \\\\\n" + \
        f"95\\% county-cluster bootstrap interval & (\\${d['boot_ci_lo']:,.0f}, \\${d['boot_ci_hi']:,.0f}) \\\\\n" + \
        r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} This is dispensary SALES revenue, not government tax revenue -- no tax rate is applied. The
benchmark multiplier $e^{-\theta}$ is applied to each CO/MT border county's own 2024 dispensary revenue and
summed, rather than to one pre-aggregated total. The bootstrap interval resamples counties with replacement
within each state (500 replicates), refitting the full interaction regression each time; it is wide and
includes zero, reflecting that $\theta$ itself is only marginally significant ($p\approx0.06$ at the 75-mile
definition) and that only three states inform this comparison. This is a model-based counterfactual
conditional on the cross-state interaction specification, not an experimentally identified fiscal-loss
estimate.
\end{minipage}
\end{table}
"""
    with open(os.path.join(OUT, "table_counterfactual.tex"), "w") as f:
        f.write(tex)
    print("Wrote table_counterfactual.tex")


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table4()
