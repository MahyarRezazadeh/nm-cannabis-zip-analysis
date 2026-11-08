"""
New Mexico border-premium estimates using the corrected, predetermined
ZIP-level border-treatment definition (nm_zip_panel_corrected.csv).

Three specifications, kept separate per the revised empirical strategy:

  1. Baseline level specification:
       log(y_zt) = alpha + beta*Border_z + gamma*log(Population_z) + delta_t + eps_zt
     beta is "the border premium": the estimated difference in the outcome
     between border and interior ZIPs, conditional on population and
     quarter fixed effects. This is NOT a claim that geography is randomly
     assigned -- see the identification discussion in the paper text.

  2. Dynamic border-premium profile (NOT a legalization event study -- the
     whole sample is post-legalization; this traces whether the border-
     interior GAP is stable or evolves as the market develops):
       log(y_zt) = alpha + beta*Border_z
                   + sum_{q != q0} beta_q*(Border_z x Quarter_q)
                   + gamma*log(Population_z) + delta_q + eps_zt
     Reference quarter q0 = 2022Q2 (first quarter of the panel).

  3. Continuous cross-border exposure (replacing the binary Border_z with a
     predetermined log(1+nonlegal population within 75mi) measure):
       log(y_zt) = alpha + beta*log(1+Exposure_z) + gamma*log(Population_z)
                   + delta_t + eps_zt

All three outcomes: revenue per capita, dispensary density (per 1,000
residents), revenue per dispensary. County-clustered standard errors
throughout (32 NM counties represented in this ZIP sample's county set).
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

OUTCOMES = [
    ("log_revenue_per_capita", "Revenue per capita"),
    ("log_dispensary_per_capita", "Dispensary density"),
    ("log_revenue_per_dispensary", "Revenue per dispensary"),
]
REF_QUARTER = "2022Q2"


def load():
    return pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"), dtype={"zip5": str, "county_fips": str})


def run_baseline(df):
    print("=== 1. Baseline border-premium specification ===")
    rows = []
    for yvar, label in OUTCOMES:
        d = df.dropna(subset=[yvar, "log_population"]).copy()
        m = smf.ols(f"{yvar} ~ border_zip + log_population + C(yq)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
        b, se, p = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
        pct = (np.exp(b) - 1) * 100
        rows.append({"outcome": label, "coef": b, "se": se, "p": p, "pct": pct,
                     "n": int(m.nobs), "n_zips": d["zip5"].nunique()})
        print(f"{label}: b={b:+.3f} ({se:.3f}), {pct:+.1f}%, p={p:.3f}, n={int(m.nobs)}, "
              f"zips={d['zip5'].nunique()}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "nm_zip_border_baseline_results.csv"), index=False)
    return res


def run_dynamic(df, yvar, label):
    d = df.dropna(subset=[yvar, "log_population"]).copy()
    quarters = sorted(d["yq"].unique())
    formula = f"{yvar} ~ border_zip*C(yq, Treatment(reference='{REF_QUARTER}')) + log_population"
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})

    rows = []
    for q in quarters:
        if q == REF_QUARTER:
            b, se, p = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
        else:
            term = f"border_zip:C(yq, Treatment(reference='{REF_QUARTER}'))[T.{q}]"
            if term not in m.params.index:
                continue
            tt = m.t_test(f"border_zip + {term}")
            b = float(np.ravel(tt.effect)[0])
            se = float(np.ravel(tt.sd)[0])
            p = float(np.ravel(tt.pvalue)[0])
        rows.append({"outcome": label, "quarter": q, "coef": b, "se": se, "p": p})
    return pd.DataFrame(rows)


def run_continuous(df):
    print("\n=== 3. Continuous cross-border exposure specification ===")
    rows = []
    for yvar, label in OUTCOMES:
        d = df.dropna(subset=[yvar, "log_population", "log_exposure"]).copy()
        m = smf.ols(f"{yvar} ~ log_exposure + log_population + C(yq)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
        b, se, p = m.params["log_exposure"], m.bse["log_exposure"], m.pvalues["log_exposure"]
        rows.append({"outcome": label, "coef": b, "se": se, "p": p, "n": int(m.nobs)})
        print(f"{label}: b={b:+.4f} ({se:.4f}), p={p:.3f}, n={int(m.nobs)}  "
              f"[log(1+nonlegal pop within 75mi)]")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "nm_zip_continuous_exposure_results.csv"), index=False)
    return res


def main():
    df = load()
    print(f"Sample: {len(df)} ZIP-quarter obs, {df.zip5.nunique()} ZIPs, "
          f"{df.county_fips.nunique()} counties, {df.yq.nunique()} quarters\n")

    run_baseline(df)

    print("\n=== 2. Dynamic border-premium profile (by quarter) ===")
    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    dyn_all = {}
    for (yvar, label), ax in zip(OUTCOMES, axes):
        ev = run_dynamic(df, yvar, label)
        dyn_all[label] = ev
        ev.to_csv(os.path.join(TBL, f"nm_zip_dynamic_borderpremium_{yvar}.csv"), index=False)
        n_sig = (ev["p"] < 0.05).sum()
        print(f"{label}: {n_sig}/{len(ev)} quarters significant at p<0.05, "
              f"coef range [{ev['coef'].min():+.2f}, {ev['coef'].max():+.2f}]")
        xt = np.arange(len(ev))
        ax.errorbar(xt, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=3,
                    markersize=4, linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(ev["quarter"], rotation=90, fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Quarter")
    axes[0].set_ylabel("Border coefficient (log points)")
    fig.suptitle("Dynamic Border Premium by Quarter, New Mexico\n"
                 "(reference quarter 2022Q2; county-clustered 95% CI)", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(OUT, "fig_nm_dynamic_borderpremium.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    run_continuous(df)


if __name__ == "__main__":
    main()
