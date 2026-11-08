"""
County-level (CO, OR, MT, NM; WA dropped) main pooled table, per explicit
user request: log(y) ~ border_county*is_capped + border_county +
state FE + quarter FE + state x quarter FE, county-clustered SEs.

Border classification: county centroid within 75mi of the nonlegal state
line, with >=1,000 nonlegal-state residents within that reach (same
methodology as the NM ZIP-code design, see build_county_border_stateline.py).
is_capped is the existing state-year entry_capped indicator (CO/MT always
1, NM always 0, OR switches 0->1 in 2022).

State FE and state x quarter FE are implemented via a single concatenated
state_period key (C(state_period)), not separate C(state)+C(period)+
C(state):C(period) terms -- verified via patsy matrix rank that the naive
three-term version is severely rank-deficient (114 of 186 columns) because
this is an UNBALANCED panel (CO starts 2015, OR 2017, MT/NM 2022): a plain
state:period interaction generates a dummy for every combination in the
cross product, including ~72 state-quarter cells with zero observations,
which breaks full rank. The concatenated-key version only creates a dummy
for state-quarter cells that actually appear in the data and is
mathematically equivalent to the same full state x quarter fixed effect.

is_capped's own standalone coefficient is dropped (confirmed via the same
rank check to be exactly collinear with state x quarter FE, since it is
constant within every state-quarter cell) -- only border_county and its
interaction with is_capped are estimated, the same fix already applied
twice earlier in this project for an identical pattern.

Revenue per store and dispensary density are reported for all 4 states but
flagged: CO's own retailer count is only genuinely time-varying from
2025Q1 onward (verified earlier this project), and OR/MT's dispensary
counts are a single current-year snapshot repeated across periods -- NM is
the only state with a fully genuine time-varying count throughout.
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
    """LaTeX-escaped version -- an unescaped % starts a comment in LaTeX and
    would silently truncate the rest of the table row."""
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def load():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    border = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county", "dist_stateline_mi"]], on="fips5", how="left")
    df["is_capped"] = df["entry_capped"]
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    print(f"{len(df)} county-quarter rows, {df.fips5.nunique()} counties, "
          f"{df.drop_duplicates('fips5').border_county.sum()} border counties")
    print(df.groupby("state_abbr")["border_county"].agg(lambda s: s.iloc[0] if s.nunique() <= 1 else "varies"))
    return df


def main_table(df):
    specs = [("Revenue per capita", df, "log_revenue_per_capita", True),
             ("Revenue per store", df, "log_revenue_per_store", True),
             ("Dispensary density", df, "log_dispensary_per_1000", True)]
    rows_tex, rows_png = [], []
    for label, d0, yvar, _ in specs:
        d = d0.dropna(subset=[yvar, "border_county", "is_capped"]).copy()
        formula = f"{yvar} ~ border_county + border_county:is_capped + C(state_period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b_b, p_b = m.params.get("border_county", np.nan), m.pvalues.get("border_county", np.nan)
        b_i, p_i = m.params.get("border_county:is_capped", np.nan), m.pvalues.get("border_county:is_capped", np.nan)
        n = int(m.nobs)
        states_used = sorted(d.state_abbr.unique())
        rows_tex.append(f"{label} & ${b_b:+.3f}$ (${fmt_pct(b_b)}$) & {p_b:.3f} & "
                         f"${b_i:+.3f}$ (${fmt_pct(b_i)}$) & {p_i:.3f} & {n} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b_b), f"{p_b:.3f}", fmt_pct_plain(b_i), f"{p_i:.3f}", n])
        print(f"{label}: n={n} ({','.join(states_used)}), border={fmt_pct_plain(b_b)} (p={p_b:.3f}), "
              f"border x capped={fmt_pct_plain(b_i)} (p={p_i:.3f})")

    tex_path = os.path.join(OUT, "table_county_border_capped_main.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{County-level border effect and entry-cap interaction (CO, OR, MT, NM; quarterly)}
\label{tab:county_border_capped_main}
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
\textit{Notes:} Sample: CO, OR, MT, NM (Washington dropped). Border = county centroid within 75 miles of the
nonlegal-state line, with $\geq$1,000 nonlegal-state residents within that reach. Capped is the state-year
entry-cap indicator (CO/MT always capped, NM always free, OR switches capped in 2022). State and state
$\times$ quarter fixed effects via a single concatenated state-quarter key (avoids phantom empty cells from
this unbalanced panel). The Capped main effect is exactly collinear with state $\times$ quarter fixed effects
and is not separately estimated (confirmed by matrix-rank check). County-clustered standard errors.
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
    ax.set_title("County-level border effect x entry-cap (CO, OR, MT, NM)", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_county_border_capped_main.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_county_border_capped_main.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    main_table(df)
