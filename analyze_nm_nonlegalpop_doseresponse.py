"""
Flexible dose-response of the New Mexico border premium to the size of the
nearby nonlegal-state population.

  log(y_zt) = alpha + sum_k beta_k * 1[NonlegalPopBin_z = k]
              + gamma*log(Population_z) + delta_t + eps_zt

NonlegalPop_z = residents of Texas/Oklahoma/Utah within 75 miles of ZIP z's
centroid (predetermined, geography only; the same measure the border rule uses).
Reference bin = ZIP codes with essentially no nonlegal population in reach.
Outcomes: log revenue per capita, log dispensary density. County-clustered SEs,
116-ZIP New Mexico panel (2022Q2-2026Q3).

Outputs:
  figures and tables/fig_nm_nonlegalpop_doseresponse.png
  figures and tables/table_nm_nonlegalpop_doseresponse.tex
  tables/nm_nonlegalpop_doseresponse_results.csv
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
sys.path.insert(0, os.path.join(BASE, "..", "Latest try", "code"))
try:
    from healy_style import apply_healy_style, PALETTE
    apply_healy_style()
    BLUE = PALETTE["blue"]
except Exception:
    BLUE = "#2166ac"

EDGES = [-1, 1_000, 50_000, 250_000, np.inf]           # thousands: 0 | (0,50] | (50,250] | 250+
LABELS = ["0", "0--50", "50--250", "250+"]              # nonlegal pop within 75 mi (000s)
MIDS = [0, 1, 2, 3]                                     # evenly spaced categorical x positions
OUTCOMES = [("log_revenue_per_capita", "Revenue per capita"),
            ("log_dispensary_per_capita", "Dispensary density")]


def load():
    d = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"),
                    dtype={"zip5": str, "county_fips": str})
    d["npop_bin"] = pd.cut(d["nonlegal_pop_within_75mi"], bins=EDGES, labels=LABELS)
    return d


def run(d, yvar):
    s = d.dropna(subset=[yvar, "log_population", "npop_bin"]).copy()
    s["npop_bin"] = s["npop_bin"].astype(str)
    m = smf.ols(f"{yvar} ~ C(npop_bin, Treatment(reference='0')) + log_population + C(yq)",
                data=s).fit(cov_type="cluster", cov_kwds={"groups": s["county_fips"]})
    rows = [{"bin": "0", "coef": 0.0, "se": 0.0, "p": np.nan}]
    for lab in LABELS[1:]:
        term = f"C(npop_bin, Treatment(reference='0'))[T.{lab}]"
        rows.append({"bin": lab, "coef": m.params[term], "se": m.bse[term], "p": m.pvalues[term]})
    r = pd.DataFrame(rows)
    r["outcome"] = yvar
    r["n"] = int(m.nobs)
    r["n_zip"] = s["zip5"].nunique()
    return r


def main():
    d = load()
    zz = d.drop_duplicates("zip5")
    print("ZIP codes per nonlegal-population bin (116-panel):")
    print(zz["npop_bin"].value_counts().sort_index().to_string())
    print()

    res = {}
    for yvar, lab in OUTCOMES:
        r = run(d, yvar)
        res[lab] = r
        print(f"--- {lab} ---")
        for _, x in r.iterrows():
            pct = (np.exp(x.coef) - 1) * 100
            tail = "" if x.bin == "0" else f"  ({x.se:.3f}), p={x.p:.3f},  {pct:+.1f}%"
            print(f"  bin {x.bin:<8} coef {x.coef:+.3f}{tail}")
        print()
    allr = pd.concat(res.values(), ignore_index=True)
    allr.to_csv(os.path.join(TBL, "nm_nonlegalpop_doseresponse_results.csv"), index=False)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (yvar, lab) in zip(axes, OUTCOMES):
        r = res[lab]
        x = np.array(MIDS)
        ax.axhline(0, color="gray", lw=0.9, ls="--")
        ax.errorbar(x, r["coef"], yerr=1.96 * r["se"], fmt="o-", color=BLUE,
                    capsize=3, markersize=5, lw=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_xlim(-0.4, 3.4)
        ax.set_title(lab, fontsize=11)
        ax.set_xlabel("Nonlegal-state population within 75 mi (000s)")
    axes[0].set_ylabel("Coefficient vs. 0 bin (log points)")
    fig.suptitle("Dose-response of the New Mexico border premium to nearby nonlegal-state population\n"
                 "reference: ZIP codes with no nonlegal population within 75 mi; county-clustered 95% CI",
                 fontsize=11)
    fig.tight_layout()
    fpath = os.path.join(OUT, "fig_nm_nonlegalpop_doseresponse.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fpath)

    # ---------- table ----------
    rc, dd = res["Revenue per capita"], res["Dispensary density"]
    def cell(x):
        if x.bin == "0":
            return "ref.\\ & ref."
        star = "".join("*" for t in (0.10, 0.05, 0.01) if x.p < t)
        return f"${x.coef:+.3f}^{{{star}}}$ & ({x.se:.3f})" if star else f"${x.coef:+.3f}$ & ({x.se:.3f})"
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Dose-response of the border premium to nearby nonlegal-state population, New Mexico}")
    lines.append(r"\label{tab:nm_nonlegalpop_doseresponse}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{Revenue per capita} & \multicolumn{2}{c}{Dispensary density} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r"Nonlegal pop.\ within 75 mi (000s) & Coef. & SE & Coef. & SE \\")
    lines.append(r"\midrule")
    for i, lab in enumerate(LABELS):
        a, b = rc.iloc[i], dd.iloc[i]
        if lab == "0":
            lines.append(f"{lab} (reference) & ref. & & ref. & \\\\")
        else:
            sa = "".join("*" for t in (0.10, 0.05, 0.01) if a.p < t)
            sb = "".join("*" for t in (0.10, 0.05, 0.01) if b.p < t)
            ca = f"${a.coef:+.3f}^{{{sa}}}$" if sa else f"${a.coef:+.3f}$"
            cb = f"${b.coef:+.3f}^{{{sb}}}$" if sb else f"${b.coef:+.3f}$"
            lines.append(f"{lab} & {ca} & ({a.se:.3f}) & {cb} & ({b.se:.3f}) \\\\")
    lines.append(r"\midrule")
    lines.append(f"Quarter fixed effects & Yes & & Yes & \\\\")
    lines.append(f"Observations & {int(rc.iloc[0].n):,} & & {int(dd.iloc[0].n):,} & \\\\")
    lines.append(f"Number of ZIP codes & {int(rc.iloc[0].n_zip)} & & {int(dd.iloc[0].n_zip)} & \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{0.92\textwidth}")
    lines.append(r"\vspace{0.4em}\footnotesize")
    lines.append(r"\textit{Notes:} Each column reports $\beta_k$ from "
                 r"$\log(y_{zt}) = \alpha + \sum_k \beta_k\,\mathbbm{1}[\text{bin}_z=k] "
                 r"+ \gamma\log(\text{Population}_z) + \delta_t + \varepsilon_{zt}$, estimated on the 116 "
                 r"New Mexico ZIP codes with any dispensary activity. Bins group ZIP codes by the "
                 r"predetermined count of Texas, Oklahoma, or Utah residents living within 75 miles of "
                 r"the ZIP centroid; the reference bin is ZIP codes with essentially none. "
                 r"$\text{Population}_z$ is the ZIP's own resident population. Quarter fixed effects, "
                 r"county-clustered standard errors. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    with open(os.path.join(OUT, "table_nm_nonlegalpop_doseresponse.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote table_nm_nonlegalpop_doseresponse.tex")


if __name__ == "__main__":
    main()
