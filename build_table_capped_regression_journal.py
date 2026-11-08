"""
Publication-standard regression table (booktabs style, SEs in parentheses,
significance stars, N/R^2/FE rows) for the border x capped-entry
specification -- log(y_cst) = alpha + beta*Border_c + Border_c*Capped_c +
delta_st + eps_cst -- at the 75-mile border definition, matching Table 8
in Appendix B (same sample: CO/OR/MT/NM, truncated to <=2025Q2,
county-clustered SEs).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def stars(p):
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""


def load():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    border = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county"]], on="fips5", how="left")
    df["is_capped"] = df["entry_capped"]
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    return df


def main():
    df = load()
    specs = [("Revenue per capita", "log_revenue_per_capita"),
             ("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]

    results = {}
    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_county", "is_capped"]).copy()
        formula = f"{yvar} ~ border_county + border_county:is_capped + C(state_period)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        results[label] = {
            "b_border": m.params["border_county"], "se_border": m.bse["border_county"],
            "p_border": m.pvalues["border_county"],
            "b_int": m.params["border_county:is_capped"], "se_int": m.bse["border_county:is_capped"],
            "p_int": m.pvalues["border_county:is_capped"],
            "n": int(m.nobs), "r2": m.rsquared, "n_counties": d.fips5.nunique(),
        }
        print(f"{label}: border={m.params['border_county']:+.3f} ({m.bse['border_county']:.3f}), "
              f"interaction={m.params['border_county:is_capped']:+.3f} ({m.bse['border_county:is_capped']:.3f}), "
              f"n={int(m.nobs)}, R2={m.rsquared:.3f}")

    labels = [l for l, _ in specs]

    def row(key_b, key_se, key_p, fmt="{:+.3f}"):
        cells = []
        for label in labels:
            r = results[label]
            b, se, p = r[key_b], r[key_se], r[key_p]
            cells.append(f"${fmt.format(b)}{stars(p)}$")
        return " & ".join(cells)

    def se_row(key_se):
        cells = []
        for label in labels:
            cells.append(f"({results[label][key_se]:.3f})")
        return " & ".join(cells)

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{County-level border premium under the entry-cap interaction: full regression results}")
    tex.append(r"\label{tab:capped_regression_journal}")
    tex.append(r"\footnotesize")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r" & (1) & (2) & (3) \\")
    tex.append(r" & Revenue per capita & Revenue per store & Dispensary density \\")
    tex.append(r"\midrule")
    tex.append(r"$\text{Border}_c$ & " + row("b_border", "se_border", "p_border") + r" \\")
    tex.append(r" & " + se_row("se_border") + r" \\")
    tex.append(r"$\text{Border}_c \times \text{Capped}_c$ & " + row("b_int", "se_int", "p_int") + r" \\")
    tex.append(r" & " + se_row("se_int") + r" \\")
    tex.append(r"\midrule")
    tex.append(r"State FE & Yes & Yes & Yes \\")
    tex.append(r"Quarter FE & Yes & Yes & Yes \\")
    tex.append(r"State $\times$ Quarter FE & Yes & Yes & Yes \\")
    tex.append(r"\midrule")
    tex.append(r"Observations & " + " & ".join(str(results[l]["n"]) for l in labels) + r" \\")
    tex.append(r"Counties & " + " & ".join(str(results[l]["n_counties"]) for l in labels) + r" \\")
    tex.append(r"$R^2$ & " + " & ".join(f"{results[l]['r2']:.3f}" for l in labels) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\begin{minipage}{0.98\textwidth}")
    tex.append(r"\vspace{0.3em}")
    tex.append(r"\footnotesize")
    tex.append(r"\textit{Notes:} Dependent variable is the log outcome named in each column heading. "
               r"Sample: Colorado, Oregon, Montana, and New Mexico county-quarter panel (Washington dropped), "
               r"truncated to 2025Q2 or earlier. $\text{Border}_c$ = 1 if county $c$'s centroid lies within 75 "
               r"miles of the nonlegal-state line, with $\geq$1,000 nonlegal-state residents within that reach. "
               r"$\text{Capped}_c$ is the state-year entry-cap indicator (Colorado and Montana capped "
               r"throughout, New Mexico free throughout, Oregon switching to capped entry in 2022); its own "
               r"main effect is exactly collinear with the state $\times$ quarter fixed effects (a state-year "
               r"variable absorbed by a strictly finer fixed effect) and is therefore not separately reported. "
               r"State and state $\times$ quarter fixed effects are implemented via a single concatenated "
               r"state-quarter key to avoid the rank deficiency a naive three-term specification produces on "
               r"this unbalanced panel. Standard errors, clustered at the county level, in parentheses. "
               r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    tex.append(r"\end{minipage}")
    tex.append(r"\end{table}")

    tex_str = "\n".join(tex)
    out_path = os.path.join(OUT, "table_capped_regression_journal.tex")
    with open(out_path, "w") as f:
        f.write(tex_str + "\n")
    print(f"\nWrote {out_path}")
    print()
    print(tex_str)


if __name__ == "__main__":
    main()
