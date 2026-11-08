"""
Dispensary-level event study for New Mexico, per explicit user request:
same border design as the paper's main ZIP-code specification
(equation \\ref{baseline} in main.tex), but applied directly to each
geocoded dispensary rather than aggregated to the ZIP level -- a
dispensary is BORDER if it sits within 75 miles of the NM state line
where NM borders a nonlegal state, provided >=1,000 nonlegal-state
residents live within that same 75-mile reach. 436 border dispensaries,
756 interior (of 1,192 geocoded; 61 of 1,253 dropped for missing
coordinates -- see build_nm_dispensary_border_classification.py).

Same event-study specification as the ZIP design:
log(y_dt) = alpha + beta*Border_d + sum_t beta_t*(Border_d x period_t)
            + gamma*log(Population_d) + delta_t + epsilon_dt
with the coefficient-LEVEL extraction fix (t_test on the linear
combination border + border:period[T.t], not the raw interaction
coefficient alone, which is a deviation from the omitted reference
period) applied at every period. County-clustered SEs (fips5).

Three outcomes -- see build_nm_dispensary_quarterly_panel.py for exact
construction and the flagged reinterpretation of "dispensary density" at
the individual-firm level (local_density, not a strict per-unit analog
of the ZIP/county-level count):
  - log_revenue_percapita: direct dispensary-level analog of the ZIP
    design's main outcome.
  - log_revenue: the ZIP design's "revenue per store" collapses to this
    at the single-dispensary level (store count = 1).
  - log_local_density: dispensaries active within 10mi per 1,000
    residents within 10mi -- an individual-location competitive-density
    measure, substituting for the market-level "dispensaries per capita"
    concept that has no literal per-firm equivalent.
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


def load():
    df = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel.csv"), dtype={"fips5": str})
    df["log_population"] = np.log(df["population"])
    # CSV round-trips -inf as the literal string "-inf" (parses back to a float
    # -inf, not NaN) for dispensary-quarters with zero sales or zero local
    # density -- must be scrubbed before dropna or it silently poisons the OLS
    # solve (produces an all-NaN fit with no error).
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"{len(df)} dispensary-quarter rows, {df.dispensary_id.nunique()} dispensaries, "
          f"{df.period.nunique()} quarters")
    print(f"Border dispensaries: {df[df.border_dispensary == 1].dispensary_id.nunique()}, "
          f"Interior: {df[df.border_dispensary == 0].dispensary_id.nunique()}")
    return df


SPECS = [("Revenue per capita", "log_revenue_percapita"),
         ("Revenue (per dispensary)", "log_revenue"),
         ("Local dispensary density", "log_local_density")]


def main_table(df):
    rows_tex, rows_png = [], []
    for label, yvar in SPECS:
        d = df.dropna(subset=[yvar, "border_dispensary", "log_population"]).copy()
        formula = f"{yvar} ~ border_dispensary + log_population + C(period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b, p, n = m.params.get("border_dispensary", np.nan), m.pvalues.get("border_dispensary", np.nan), int(m.nobs)
        n_treat = d[d.border_dispensary == 1].dispensary_id.nunique()
        n_ctrl = d[d.border_dispensary == 0].dispensary_id.nunique()
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & {n} & {n_treat} & {n_ctrl} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b), f"{p:.3f}", n, n_treat, n_ctrl])
        print(f"{label}: n={n} (treat={n_treat}, ctrl={n_ctrl}), border effect={fmt_pct_plain(b)} "
              f"(b={b:+.3f}, p={p:.3f})")

    tex_path = os.path.join(OUT, "table_nm_dispensary_border_main.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{New Mexico dispensary-level border effect (75-mile state-line definition)}
\label{tab:nm_dispensary_border_main}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Outcome & Coef. (\%) & $p$ (cl.) & $n$ & Border disp. & Interior disp. \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.95\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Sample: geocoded New Mexico dispensaries with valid coordinates (1,192 of 1,253).
Border = dispensary within 75 miles of the nonlegal-state line, with $\geq$1,000 nonlegal-state
residents within that reach -- the same rule as the paper's main ZIP-code design, applied directly
to each dispensary's own coordinates rather than aggregated to the ZIP level. County population
control (in logs), quarter fixed effects, county-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")

    fig, ax = plt.subplots(figsize=(10, 0.6 * (len(rows_png) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows_png,
                    colLabels=["Outcome", "Border effect", "p (cl.)", "n", "Border disp.", "Interior disp."],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    ax.set_title("NM dispensary-level border effect (75mi state-line definition)", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_nm_dispensary_border_main.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_nm_dispensary_border_main.png")


def event_study(df):
    es_results = {}
    for label, yvar in SPECS:
        d = df.dropna(subset=[yvar, "border_dispensary", "log_population"]).copy()
        formula = f"{yvar} ~ border_dispensary + border_dispensary:C(period) + log_population + C(period)"
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
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_dispensary_{label.replace(' ', '').replace('(', '').replace(')', '')}.csv"),
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
    fig.suptitle("Border x quarter event study, NM dispensary level (75mi state-line definition)\n"
                  "log-population control, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_dispensary.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    main_table(df)
    event_study(df)
