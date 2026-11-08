"""
Monthly and quarterly re-analysis of the border x entry-cap design, with
Washington dropped from the sample per explicit user request (leaving CO,
OR, MT, NM). Uses master_monthly_panel_no_wa.csv / master_quarterly_panel_no_wa.csv
from build_monthly_quarterly_panels.py.

Revenue per capita is estimated for all 4 states (genuine monthly/quarterly
revenue everywhere). Revenue per store and dispensary density are estimated
CO+NM only, since OR and MT's dispensary counts are a single current-year
snapshot repeated across periods in this project (no historical count
exists), and a "per-store"/density outcome built on a constant denominator
would not carry genuine within-state time variation -- reported honestly as
a data-availability limit ("to the extent possible"), not silently forced.

Produces, for each frequency (monthly, quarterly):
  1. Main pooled border x capped interaction table (pooled regime sample)
  2. Border x period event-study figure (pooled regime sample)
  3. Border-only main table, split into capped-only and free-only subsamples
  4. Border x period event-study figure, split into capped-only and free-only

All tables saved as both .tex and .png; all figures as .png; everything
under figures and tables/.
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

STORE_DENSITY_STATES = ["CO", "NM"]  # only states with genuinely time-varying dispensary counts


def load(freq):
    fname = "master_monthly_panel_no_wa.csv" if freq == "monthly" else "master_quarterly_panel_no_wa.csv"
    df = pd.read_csv(os.path.join(TBL, fname), dtype={"fips5": str})
    period_col = "ym" if freq == "monthly" else "yq"
    df["period"] = df[period_col]
    return df


def outcomes_for(df, which):
    """Return (label, subset_df, yvar) tuples available at this frequency."""
    specs = []
    d = df.dropna(subset=["log_revenue_per_capita", "log_population"]).copy()
    specs.append(("Revenue per capita", d, "log_revenue_per_capita"))
    if which in ("store", "both"):
        d2 = df[(df.state_abbr.isin(STORE_DENSITY_STATES)) & (df.count_is_snapshot == 0)].dropna(
            subset=["log_revenue_per_store", "log_population"]).copy()
        specs.append(("Revenue per store", d2, "log_revenue_per_store"))
        d3 = df[(df.state_abbr.isin(STORE_DENSITY_STATES)) & (df.count_is_snapshot == 0)].dropna(
            subset=["log_dispensary_per_1000", "log_population"]).copy()
        specs.append(("Dispensary density", d3, "log_dispensary_per_1000"))
    return specs


def save_table_png(rows, col_labels, path_png, title):
    fig, ax = plt.subplots(figsize=(min(14, 1.4 * len(col_labels) + 2), 0.6 * (len(rows) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title(title, fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def main_pooled_table(df, freq):
    specs = outcomes_for(df, "both")
    rows_tex, rows_png = [], []
    for label, d, yvar in specs:
        n_states = d.state_abbr.nunique()
        # Revenue-per-store / dispensary-density are CO+NM only, and entry_capped is exactly
        # collinear with state (CO always capped, NM always free) -- including state FE there
        # would make entry_capped and the interaction rank-deficient (arbitrary pinv values),
        # the same pitfall already documented in the main draft's Table~\ref{tab:result1} notes.
        # State FE is therefore omitted for these two outcomes, matching that precedent exactly.
        use_state_fe = n_states > 1 and label == "Revenue per capita"
        formula = f"{yvar} ~ border_adjacent * entry_capped + log_population + C(period)" + \
                  (" + C(state_abbr)" if use_state_fe else "")
        m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b_border = m_cl.params.get("border_adjacent", np.nan)
        p_border = m_cl.pvalues.get("border_adjacent", np.nan)
        b_cap = m_cl.params.get("entry_capped", np.nan)
        p_cap = m_cl.pvalues.get("entry_capped", np.nan)
        b_int = m_cl.params.get("border_adjacent:entry_capped", np.nan)
        p_int = m_cl.pvalues.get("border_adjacent:entry_capped", np.nan)
        n = int(m_cl.nobs)
        rows_tex.append(f"{label} & ${b_border:+.3f}$ (${fmt_pct(b_border)}$) & {p_border:.3f} & "
                         f"${b_cap:+.3f}$ (${fmt_pct(b_cap)}$) & {p_cap:.3f} & "
                         f"${b_int:+.3f}$ (${fmt_pct(b_int)}$) & {p_int:.3f} & {n} \\\\\n")
        rows_png.append([label, f"{fmt_pct(b_border)} (p={p_border:.3f})", f"{fmt_pct(b_cap)} (p={p_cap:.3f})",
                          f"{fmt_pct(b_int)} (p={p_int:.3f})", n])
        print(f"[{freq}] {label}: n={n}, states={n_states}, border={fmt_pct(b_border)} (p={p_border:.3f}), "
              f"capped={fmt_pct(b_cap)} (p={p_cap:.3f}), interaction={fmt_pct(b_int)} (p={p_int:.3f})")

    tex_path = os.path.join(OUT, f"table_main_pooled_{freq}_noWA.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Main pooled specification, """ + freq + r""" panel, Washington dropped}
\label{tab:main_pooled_""" + freq + r"""_noWA}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
& \multicolumn{2}{c}{Border} & \multicolumn{2}{c}{Capped} & \multicolumn{2}{c}{Border $\times$ Capped} & \\
Outcome & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) & $n$ \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Sample: CO, OR, MT, NM (Washington dropped). Border is county-to-state-line geometric
adjacency, entry-cap classification as in the main design. Period fixed effects are """ + \
    ("calendar year-month" if freq == "monthly" else "calendar year-quarter") + \
    r""" dummies. Revenue per store and dispensary density are estimated for Colorado and New Mexico only, the
only two states with a genuinely time-varying (not single-snapshot) dispensary count at this frequency.
County-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"Wrote {tex_path}")

    save_table_png(rows_png, ["Outcome", "Border", "Capped", "Border x Capped", "n"],
                    os.path.join(OUT, f"table_main_pooled_{freq}_noWA.png"),
                    f"Main pooled specification ({freq}, no WA)")


def event_study(df, freq, regime_label=None, regime_filter=None):
    d0 = df if regime_filter is None else df[regime_filter(df)]
    specs = outcomes_for(d0, "both")
    tag = f"{freq}" if regime_label is None else f"{freq}_{regime_label}"
    es_results = {}
    for label, d, yvar in specs:
        d = d.dropna(subset=[yvar, "border_adjacent", "log_population"]).copy()
        n_states = d.state_abbr.nunique()
        if d["border_adjacent"].nunique() < 2 or len(d) < 30:
            print(f"[{tag}] {label}: skipped (insufficient variation, n={len(d)})")
            continue
        formula = f"{yvar} ~ border_adjacent + border_adjacent:C(period) + log_population + C(period)" + \
                  (" + C(state_abbr)" if n_states > 1 else "")
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        # Each C(period) interaction term is a DEVIATION from the omitted reference period, not
        # the period's own level -- the true period-k border effect is border_adjacent +
        # border_adjacent:C(period)[T.k] (border_adjacent alone at the omitted reference period).
        # Verified against a raw, model-free border-vs-interior mean-difference check before fixing.
        all_periods = sorted(d["period"].unique())
        interacted = {t.split("[T.")[1].rstrip("]") for t in m.params.index
                      if t.startswith("border_adjacent:C(period)")}
        rows = []
        for p in all_periods:
            if p in interacted:
                test_str = f"border_adjacent + border_adjacent:C(period)[T.{p}] = 0"
            else:
                test_str = "border_adjacent = 0"
            tt = m.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        es_results[label] = pd.DataFrame(rows, columns=["period", "coef", "se"])
        es_results[label].to_csv(os.path.join(OUT, f"eventstudy_{tag}_{label.replace(' ', '')}.csv"), index=False)
        print(f"[{tag}] {label}: n={len(d)}, {len(all_periods)} period coefficients estimated")

    if not es_results:
        print(f"[{tag}] no outcomes had enough variation for an event study -- skipping figure")
        return

    apply_healy_style()
    n = len(es_results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]
    for ax, (label, dd) in zip(axes, es_results.items()):
        step = max(1, len(dd) // 12)
        xt = np.arange(len(dd))
        ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=2, markersize=3,
                    linewidth=1, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt[::step])
        ax.set_xticklabels(dd["period"].iloc[::step], rotation=90, fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Border coefficient (log points)")
    title = f"Border x period event study ({freq}, no WA)" if regime_label is None else \
        f"Border x period event study ({freq}, no WA) -- {regime_label} regime"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, f"fig_eventstudy_{tag}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def regime_split_table(df, freq):
    rows_tex, rows_png = [], []
    for regime_label, cap_val in [("Capped", 1), ("Free", 0)]:
        sub = df[df.entry_capped == cap_val]
        specs = outcomes_for(sub, "both")
        for label, d, yvar in specs:
            n_states = d.state_abbr.nunique()
            if d["border_adjacent"].nunique() < 2 or len(d) < 20:
                print(f"[{freq}/{regime_label}] {label}: skipped (n={len(d)}, insufficient variation)")
                continue
            formula = f"{yvar} ~ border_adjacent + log_population + C(period)" + \
                      (" + C(state_abbr)" if n_states > 1 else "")
            m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
            b = m_cl.params.get("border_adjacent", np.nan)
            p = m_cl.pvalues.get("border_adjacent", np.nan)
            n = int(m_cl.nobs)
            states = sorted(d.state_abbr.unique())
            rows_tex.append(f"{regime_label} & {label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & {n} & "
                             f"{','.join(states)} \\\\\n")
            rows_png.append([regime_label, label, f"{fmt_pct(b)} (p={p:.3f})", n, ",".join(states)])
            print(f"[{freq}/{regime_label}] {label}: n={n} ({','.join(states)}), border={fmt_pct(b)} (p={p:.3f})")

    tex_path = os.path.join(OUT, f"table_regime_split_{freq}_noWA.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Border effect by entry regime, """ + freq + r""" panel, Washington dropped}
\label{tab:regime_split_""" + freq + r"""_noWA}
\footnotesize
\begin{tabular}{llcccl}
\toprule
Regime & Outcome & Border Coef. (\%) & $p$ (cl.) & $n$ & States contributing \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Sample split by the state-year entry-cap indicator, not estimated jointly with an interaction
term. CO and MT are capped for their entire panel window and so contribute nothing to the Free row; New
Mexico is free-entry for its entire panel window and so contributes nothing to the Capped row; Oregon
contributes to both (free 2015--2021, capped 2022 onward). County-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"Wrote {tex_path}")

    save_table_png(rows_png, ["Regime", "Outcome", "Border", "p (cl.)", "n", "States"],
                    os.path.join(OUT, f"table_regime_split_{freq}_noWA.png"),
                    f"Border effect by entry regime ({freq}, no WA)")


def main():
    os.makedirs(OUT, exist_ok=True)
    for freq in ["monthly", "quarterly"]:
        print("\n" + "=" * 80)
        print(f"FREQUENCY: {freq}")
        print("=" * 80)
        df = load(freq)
        print(f"Loaded {len(df)} rows, states={sorted(df.state_abbr.unique())}, "
              f"periods {df['period'].min()}-{df['period'].max()}")

        main_pooled_table(df, freq)
        event_study(df, freq)
        regime_split_table(df, freq)
        event_study(df, freq, regime_label="Capped", regime_filter=lambda d: d.entry_capped == 1)
        event_study(df, freq, regime_label="Free", regime_filter=lambda d: d.entry_capped == 0)


if __name__ == "__main__":
    main()
