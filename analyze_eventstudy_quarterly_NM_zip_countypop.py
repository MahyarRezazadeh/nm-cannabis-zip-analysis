"""
NM ZIP-code border design, revised per explicit user request:
  - C(state_abbr) and C(state_abbr):C(quarter) REMOVED (both were no-ops on
    this NM-only panel anyway, fully subsumed by C(period)).
  - The two population controls (ZIP's own population, population within
    70mi) REPLACED by a single county-population control: the population of
    the COUNTY the ZIP code falls in, from the same NM county population
    figures used throughout the rest of this project (master_percapita_panel.csv,
    a single Census estimate per county, constant across years).

Specification: log(y) ~ border_zip + border_zip:C(period) + log_county_population
+ C(period), clustered by county.

Produces both the main pooled table (one border coefficient per outcome)
and the border x quarter event-study figure.
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

    countypop = pd.read_csv(os.path.join(SOURCE_BASE, "tables", "master_percapita_panel.csv"),
                             dtype={"fips5": str})
    countypop = countypop[countypop.state_abbr == "NM"][["fips5", "population"]].drop_duplicates()
    countypop = countypop.rename(columns={"fips5": "county_fips_clean", "population": "county_population"})

    before = df.zip5.nunique()
    df = df.merge(countypop, on="county_fips_clean", how="left")
    unmatched = df[df.county_population.isna()][["zip5", "county_fips_clean"]].drop_duplicates()
    if len(unmatched):
        print(f"ZIP codes with no matching NM county population (dropped -- likely out-of-state county_fips, "
              f"e.g. the El Paso, TX zip): \n{unmatched}")
    df["log_county_population"] = np.log(df["county_population"])
    print(f"{before} ZIP codes in panel, {df.dropna(subset=['county_population']).zip5.nunique()} "
          f"matched to an NM county population")
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
        b = m.params.get("border_zip", np.nan)
        p = m.pvalues.get("border_zip", np.nan)
        n = int(m.nobs)
        rows_tex.append(f"{label} & ${b:+.3f}$ (${fmt_pct(b)}$) & {p:.3f} & {n} \\\\\n")
        rows_png.append([label, fmt_pct(b), f"{p:.3f}", n])
        print(f"{label}: n={n}, border effect={fmt_pct(b)} (b={b:+.3f}, p={p:.3f})")

    tex_path = os.path.join(OUT, "table_nm_zip_countypop.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{New Mexico ZIP-code border effect, county-population control}
\label{tab:nm_zip_countypop}
\footnotesize
\begin{tabular}{lccc}
\toprule
Outcome & Coef. (\%) & $p$ (cl.) & $n$ \\
\midrule
""")
        f.writelines(rows_tex)
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.9\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Border = ZIP code within 50 miles of the nearest incorporated place (population $\geq$5,000)
in a state with no operating recreational retail market; interior otherwise. Population control is the
population of the county the ZIP code falls in (a single Census estimate per county, constant across the
panel), replacing both the ZIP's own population and the 70-mile catchment population used in earlier
versions. County-clustered standard errors.
\end{minipage}
\end{table}
""")
    print(f"\nWrote {tex_path}")

    fig, ax = plt.subplots(figsize=(8, 0.6 * (len(rows_png) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows_png, colLabels=["Outcome", "Border effect", "p (cl.)", "n"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    ax.set_title("NM ZIP-code border effect: county-population control", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_nm_zip_countypop.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote table_nm_zip_countypop.png")


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
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_zip_countypop_{label.replace(' ', '')}.csv"),
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
    fig.suptitle("Border x quarter event study, NM ZIP panel -- county-population control\n"
                  "county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_zip_countypop.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    main_table(df)
    event_study(df)
