"""
City-level event study for OR and MT dispensary density, per explicit
user request. Same border rule as the rest of this project (75mi state-
line + >=1,000 nonlegal population). Only outcome available is dispensary
density (dispensaries per 1,000 residents) -- neither state publishes
revenue below the county level, so no revenue outcome exists at the city
level for either (see build scripts for full data-availability notes).

Both panels are STATIC SNAPSHOTS repeated across quarters (disclosed in
the build scripts, not fabricated time variation), so this event study
is really testing whether a level difference between border and interior
cities is stable across the quarter fixed effects, not genuine dynamics
-- reported as such, not oversold as a dynamic effect.

Specification: log_density ~ border_city + border_city:C(period) +
C(period), city-clustered SEs (each city is its own cluster; no higher-
level geography is available in this panel). Level at each period
extracted via t_test (border_city + border_city:C(period)[T.k]), the
same fix used throughout this project, applied at every period including
the omitted reference period.
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


def fmt_pct_plain(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def run_state(state_abbr, panel_file, border_file):
    df = pd.read_csv(os.path.join(TBL, panel_file))
    border = pd.read_csv(os.path.join(TBL, border_file))
    df = df.merge(border[["city", "border_city", "dist_stateline_mi", "nonlegal_pop_within_75mi"]],
                  on="city", how="left")
    df["period"] = df["yq"]
    df = df.dropna(subset=["log_density", "border_city"]).copy()

    n_border = df.drop_duplicates("city").border_city.sum()
    n_total = df.city.nunique()
    print(f"[{state_abbr}] {len(df)} city-quarter rows, {n_total} cities "
          f"({n_border} border, {n_total - n_border} interior), {df.period.nunique()} quarters")

    # ---------- pooled table ----------
    m = smf.ols("log_density ~ border_city + C(period)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["city"]})
    b, p, n = m.params["border_city"], m.pvalues["border_city"], int(m.nobs)
    print(f"[{state_abbr}] pooled: border={fmt_pct_plain(b)} (b={b:+.3f}, p={p:.3f}), n={n}")

    tex_path = os.path.join(OUT, f"table_{state_abbr.lower()}_city_density.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{""" + state_abbr + r""" city-level dispensary density, border vs.\ interior}
\label{tab:""" + state_abbr.lower() + r"""_city_density}
\footnotesize
\begin{tabular}{lccccc}
\toprule
Outcome & Coef. (\%) & $p$ (cl.) & $n$ & Border cities & Interior cities \\
\midrule
Dispensary density & $""" + f"{b:+.3f}" + r"""$ ($""" + fmt_pct(b) + r"""$) & """ + f"{p:.3f}" + r""" & """ + \
                str(n) + r""" & """ + str(int(n_border)) + r""" & """ + str(int(n_total - n_border)) + r""" \\
\bottomrule
\end{tabular}
\begin{minipage}{0.95\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Border = city centroid within 75 miles of the nonlegal-state line, with $\geq$1,000
nonlegal-state residents within that reach. Panel is a static cross-sectional snapshot of currently
""" + ("licensed" if state_abbr == "OR" else "listed") + r""" dispensaries repeated across every quarter
(no dispensary-level date information is available to build genuine time variation for """ + state_abbr + \
                r"""); no revenue outcome exists at this geography since """ + state_abbr + r""" publishes sales
only at the county level. City-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"Wrote {tex_path}")

    # ---------- event study ----------
    formula = "log_density ~ border_city + border_city:C(period) + C(period)"
    m2 = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["city"]})
    all_periods = sorted(df["period"].unique(), key=lambda s: (int(s[:4]), int(s[-1])))
    interacted = {t.split("[T.")[1].rstrip("]") for t in m2.params.index
                  if t.startswith("border_city:C(period)")}
    rows = []
    for p_ in all_periods:
        test_str = (f"border_city + border_city:C(period)[T.{p_}] = 0" if p_ in interacted
                    else "border_city = 0")
        tt = m2.t_test(test_str)
        rows.append((p_, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
    dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
    dd.to_csv(os.path.join(OUT, f"eventstudy_{state_abbr.lower()}_city_density.csv"), index=False)

    apply_healy_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    step = max(1, len(dd) // 20)
    xt = np.arange(len(dd))
    ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                linewidth=1.2, color=PALETTE["blue"])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(xt[::step])
    ax.set_xticklabels(dd["period"].iloc[::step], rotation=90, fontsize=8)
    ax.set_title(f"{state_abbr} city-level border x quarter event study (dispensary density)\n"
                 f"static snapshot repeated across quarters, city-clustered 95% CI", fontsize=11)
    ax.set_ylabel("Border coefficient (log points)")
    fig.tight_layout()
    out_path = os.path.join(OUT, f"fig_eventstudy_{state_abbr.lower()}_city_density.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    run_state("OR", "or_city_quarterly_panel.csv", "or_city_border_classification.csv")
    run_state("MT", "mt_city_quarterly_panel.csv", "mt_city_border_classification.csv")
