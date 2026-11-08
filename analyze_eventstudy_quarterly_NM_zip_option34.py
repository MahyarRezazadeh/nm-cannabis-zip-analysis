"""
NM ZIP-code event study, expanded treatment definition per explicit user
request: a ZIP is border if ANY of its geocoded dispensaries (not a
majority) sits within 75 miles of the nonlegal-state LINE (not a
qualifying city), AND at least 1,000 nonlegal-state residents live within
that 75-mile reach (a floor against counting genuinely empty stretches of
border as "close"). 38 border ZIP codes in the regression sample (vs. 18 in
the prior geocoded/majority-vote/city-distance design).

Same downstream specification as the kept design otherwise: county-
population control, quarter FE, county-clustered SEs.
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
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def load():
    df = pd.read_csv(os.path.join(TBL, "nm_zip_quarterly_panel.csv"), dtype={"zip5": str, "county_fips": str})
    df["period"] = df["yq"]
    df["county_fips_clean"] = df["county_fips"].str.replace(r"\.0$", "", regex=True).str.zfill(5)

    opt34 = pd.read_csv(os.path.join(TBL, "nm_zip_border_option34.csv"), dtype={"zip5": str})
    df = df.drop(columns=["border_zip"]).merge(opt34[["zip5", "border_zip"]], on="zip5", how="left")
    df["border_zip"] = df["border_zip"].fillna(0).astype(int)

    countypop = pd.read_csv(os.path.join(SOURCE_BASE, "tables", "master_percapita_panel.csv"), dtype={"fips5": str})
    countypop = countypop[countypop.state_abbr == "NM"][["fips5", "population"]].drop_duplicates()
    countypop = countypop.rename(columns={"fips5": "county_fips_clean", "population": "county_population"})
    df = df.merge(countypop, on="county_fips_clean", how="left")
    df["log_county_population"] = np.log(df["county_population"])
    matched = df.dropna(subset=["county_population"])
    print(f"{df.zip5.nunique()} ZIP codes, {matched.zip5.nunique()} matched to an NM county population, "
          f"{matched.drop_duplicates('zip5').border_zip.sum()} classified border")
    return df


def main_table(df):
    specs = [("Revenue per capita", "log_revenue_per_capita"),
             ("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]
    rows_tex, rows_png = [], []
    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_zip", "log_county_population"]).copy()
        formula = f"{yvar} ~ border_zip + log_county_population + C(period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips_clean"]})
        b, p, n = m.params.get("border_zip", np.nan), m.pvalues.get("border_zip", np.nan), int(m.nobs)
        n_treat = d[d.border_zip == 1].zip5.nunique()
        n_ctrl = d[d.border_zip == 0].zip5.nunique()
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & {n} & {n_treat} & {n_ctrl} \\\\\n")
        rows_png.append([label, fmt_pct(b), f"{p:.3f}", n, n_treat, n_ctrl])
        print(f"{label}: n={n} (treat={n_treat}, ctrl={n_ctrl}), border effect={fmt_pct(b)} (b={b:+.3f}, p={p:.3f})")

    tex_path = os.path.join(OUT, "table_nm_zip_option34.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{New Mexico ZIP-code border effect, expanded (state-line, any-dispensary, 75mi) classification}
\label{tab:nm_zip_option34}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Outcome & Coef. (\%) & $p$ (cl.) & $n$ & Border ZIPs & Interior ZIPs \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.95\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Border = ZIP code with at least one geocoded dispensary within 75 miles of the nonlegal-state
line (not a qualifying city), provided at least 1,000 nonlegal-state residents live within that 75-mile reach.
County-population control, county-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")

    fig, ax = plt.subplots(figsize=(10, 0.6 * (len(rows_png) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows_png, colLabels=["Outcome", "Border effect", "p (cl.)", "n", "Border ZIPs", "Interior ZIPs"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    ax.set_title("NM ZIP border effect: expanded (state-line, 75mi) classification", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_nm_zip_option34.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_nm_zip_option34.png")


def event_study(df):
    specs = [("Revenue per capita", "log_revenue_per_capita"),
             ("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]
    es_results = {}
    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_zip", "log_county_population"]).copy()
        formula = f"{yvar} ~ border_zip + border_zip:C(period) + log_county_population + C(period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips_clean"]})

        all_periods = sorted(d["period"].unique())
        interacted = {t.split("[T.")[1].rstrip("]") for t in m.params.index
                      if t.startswith("border_zip:C(period)")}
        rows = []
        for p in all_periods:
            test_str = (f"border_zip + border_zip:C(period)[T.{p}] = 0" if p in interacted
                        else "border_zip = 0")
            tt = m.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        es_results[label] = dd
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_zip_option34_{label.replace(' ', '')}.csv"),
                   index=False)
        print(f"{label}: n={len(d)}, {len(all_periods)} quarters, "
              f"border main (ref. period) effect b={m.params.get('border_zip', float('nan')):+.3f} "
              f"p={m.pvalues.get('border_zip', float('nan')):.3f}")

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
    fig.suptitle("Border x quarter event study, NM -- expanded (state-line, 75mi, any-dispensary) classification\n"
                  "county-population control, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_zip_option34.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    main_table(df)
    event_study(df)
