"""
v2 of the dispensary-level event study, addressing noise sources
diagnosed in _diagnose_dispensary_noise.py, per explicit user request:

1. local_density: 10mi -> 25mi radius (fixes a badly right-skewed ratio
   from near-zero population denominators; see build_nm_dispensary_
   quarterly_panel_v2.py). Max ratio drops from 97.1 to 8.2.

2. Chain/brand fixed effects (C(brand), mechanically extracted from
   dispensary_label). Full rank, but only 56 of 684 brands have BOTH
   border and interior locations -- only those (31.2% of the sample)
   actually identify beta once brand FE is added; the rest are absorbed
   entirely into their own brand's fixed effect. Reported as a SEPARATE
   table alongside the no-brand-FE main spec, not silently substituted,
   given that real cost.

4. Population control: county population -> population within the same
   25-mile radius used for local_density (log_local_population), for
   ALL three outcomes -- a much tighter proxy for each dispensary's
   actual local market than a single county-wide figure.

5. WLS weighting by dispensary age in quarters (capped at 8 quarters /
   2 years), to down-weight brand-new, still-ramping-up dispensary-
   quarters relative to established stores.

Plus: standard errors clustered at the ZIP level (zip5_nearest, each
dispensary's nearest ZIP centroid) instead of the county level.
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
AGE_CAP = 8


def fmt_pct_plain(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}\\%"


def load():
    df = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v2.csv"), dtype={"fips5": str, "zip5_nearest": str})
    df = df.replace([np.inf, -np.inf], np.nan)
    df["wt"] = (df["dispensary_age_q"] + 1).clip(upper=AGE_CAP)
    print(f"{len(df)} dispensary-quarter rows, {df.dispensary_id.nunique()} dispensaries, "
          f"{df.zip5_nearest.nunique()} ZIP clusters")
    return df


SPECS = [("Revenue per capita", "log_revenue_percapita"),
         ("Revenue (per dispensary)", "log_revenue"),
         ("Local dispensary density", "log_local_density")]


def main_table(df):
    rows_tex, rows_png = [], []
    for label, yvar in SPECS:
        d = df.dropna(subset=[yvar, "border_dispensary", "log_local_population", "wt"]).copy()
        formula = f"{yvar} ~ border_dispensary + log_local_population + C(period)"
        m = smf.wls(formula, data=d, weights=d["wt"]).fit(cov_type="cluster", cov_kwds={"groups": d["zip5_nearest"]})
        b, p, n = m.params.get("border_dispensary", np.nan), m.pvalues.get("border_dispensary", np.nan), int(m.nobs)
        n_treat = d[d.border_dispensary == 1].dispensary_id.nunique()
        n_ctrl = d[d.border_dispensary == 0].dispensary_id.nunique()
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & {n} & {n_treat} & {n_ctrl} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b), f"{p:.3f}", n, n_treat, n_ctrl])
        print(f"[main v2] {label}: n={n} (treat={n_treat}, ctrl={n_ctrl}), border effect={fmt_pct_plain(b)} "
              f"(b={b:+.3f}, p={p:.3f})")

    tex_path = os.path.join(OUT, "table_nm_dispensary_border_main_v2.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{New Mexico dispensary-level border effect, noise-reduction specification}
\label{tab:nm_dispensary_border_main_v2}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Outcome & Coef. (\%) & $p$ (cl.) & $n$ & Border disp. & Interior disp. \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Same border definition as Table~\ref{tab:nm_dispensary_border_main}. Four changes from that
baseline: (i) local dispensary density uses a 25-mile radius rather than 10 miles, correcting a right-skewed
ratio driven by near-zero population denominators (max ratio falls from 97.1 to 8.2); (ii) the population
control is population within that same 25-mile radius rather than county population; (iii) observations are
weighted by dispensary age in quarters (capped at 8) via WLS, down-weighting brand-new, still-ramping-up
dispensary-quarters; (iv) standard errors are clustered at the ZIP level (each dispensary's nearest ZIP
centroid) rather than the county level.
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
    ax.set_title("NM dispensary-level border effect: noise-reduction spec", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_nm_dispensary_border_main_v2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_nm_dispensary_border_main_v2.png")


def brand_fe_table(df):
    rows_tex, rows_png = [], []
    for label, yvar in SPECS:
        d = df.dropna(subset=[yvar, "border_dispensary", "log_local_population", "wt", "brand"]).copy()
        mix = d.groupby("brand")["border_dispensary"].nunique()
        mixed = set(mix[mix > 1].index)
        n_mixed_obs = d[d.brand.isin(mixed)].shape[0]
        formula = f"{yvar} ~ border_dispensary + log_local_population + C(period) + C(brand)"
        m = smf.wls(formula, data=d, weights=d["wt"]).fit(cov_type="cluster", cov_kwds={"groups": d["zip5_nearest"]})
        b, p, n = m.params.get("border_dispensary", np.nan), m.pvalues.get("border_dispensary", np.nan), int(m.nobs)
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & {n} & {len(mixed)} & "
                         f"{n_mixed_obs} \\\\\n")
        rows_png.append([label, fmt_pct_plain(b), f"{p:.3f}", n, len(mixed), n_mixed_obs])
        print(f"[brand FE] {label}: n={n}, mixed brands={len(mixed)}, identifying obs={n_mixed_obs} "
              f"({n_mixed_obs/n*100:.1f}%), border effect={fmt_pct_plain(b)} (b={b:+.3f}, p={p:.3f})")

    tex_path = os.path.join(OUT, "table_nm_dispensary_border_brandfe.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{New Mexico dispensary-level border effect, with chain/brand fixed effects}
\label{tab:nm_dispensary_border_brandfe}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Outcome & Coef. (\%) & $p$ (cl.) & $n$ & Mixed brands & Identifying $n$ \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Same specification as Table~\ref{tab:nm_dispensary_border_main_v2}, adding a chain/brand fixed
effect mechanically extracted from each dispensary's business name. Brand fixed effects fully absorb the
border indicator for any chain operating only border or only interior locations; the border coefficient is
therefore identified only from ``mixed'' chains that operate both. ``Identifying $n$'' reports how many of the
$n$ observations belong to a mixed chain (roughly 31 percent in every outcome) -- a real precision cost from
adding brand fixed effects, reported here rather than left implicit.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")


def event_study(df):
    es_results = {}
    for label, yvar in SPECS:
        d = df.dropna(subset=[yvar, "border_dispensary", "log_local_population", "wt"]).copy()
        formula = f"{yvar} ~ border_dispensary + border_dispensary:C(period) + log_local_population + C(period)"
        m = smf.wls(formula, data=d, weights=d["wt"]).fit(cov_type="cluster", cov_kwds={"groups": d["zip5_nearest"]})

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
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_dispensary_v2_{label.replace(' ', '').replace('(', '').replace(')', '')}.csv"),
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
    fig.suptitle("Border x quarter event study, NM dispensary level -- noise-reduction spec\n"
                  "local-population control, age-weighted WLS, ZIP-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_dispensary_v2.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    main_table(df)
    brand_fe_table(df)
    event_study(df)
