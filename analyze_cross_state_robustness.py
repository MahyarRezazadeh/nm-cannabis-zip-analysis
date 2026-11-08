"""
Robustness checks for the three-state (NM, CO, MT) cross-regime comparison:
  1. Distance-threshold robustness (50/75/100mi) for the pooled Border x
     Restricted interaction, principal outcome = revenue per capita.
  2. Continuous cross-border exposure interaction:
       log(y_cst) = alpha + beta*Exposure_c + theta*(Exposure_c x Restricted_s)
                     + delta_st + eps_cst
     Exposure_c = log(1 + nonlegal population within 75 miles), predetermined
     county-level measure already used elsewhere in this project.

Oregon is excluded throughout (three-state sample: NM, CO, MT only).
County-clustered standard errors. Same state-quarter fixed-effect
implementation (concatenated key) as the rest of the cross-state analysis.
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

STATES = ["NM", "CO", "MT"]
RESTRICTED = {"NM": 0, "CO": 1, "MT": 1}
THRESHOLDS = {50: "county_border_stateline_50mi.csv", 75: "county_border_stateline.csv",
              100: "county_border_stateline_100mi.csv"}
POPCOL = {50: "nonlegal_pop_within_50mi", 75: "nonlegal_pop_within_75mi", 100: "nonlegal_pop_within_100mi"}


def load_base():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[df.state_abbr.isin(STATES)].copy()
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    df["restricted"] = df["state_abbr"].map(RESTRICTED)
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    return df


def threshold_robustness(df_base):
    print("=== 1. Distance-threshold robustness (revenue per capita, Border x Restricted) ===")
    rows = []
    for mi, fname in THRESHOLDS.items():
        border = pd.read_csv(os.path.join(TBL, fname), dtype={"fips5": str})
        df = df_base.merge(border[["fips5", "border_county"]], on="fips5", how="left")
        d = df.dropna(subset=["log_revenue_per_capita", "border_county", "restricted"]).copy()
        m = smf.ols("log_revenue_per_capita ~ border_county + border_county:restricted + C(state_period)",
                     data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b = m.params.get("border_county:restricted", np.nan)
        se = m.bse.get("border_county:restricted", np.nan)
        p = m.pvalues.get("border_county:restricted", np.nan)
        n_border = d.drop_duplicates("fips5").border_county.sum()
        rows.append({"threshold_mi": mi, "coef": b, "se": se, "p": p, "n": int(m.nobs),
                     "n_border_counties": int(n_border)})
        print(f"{mi}mi: border x restricted = {(np.exp(b)-1)*100:+.1f}% (b={b:+.3f}, se={se:.3f}), "
              f"p={p:.3f}, n={int(m.nobs)}, border counties={int(n_border)}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "cross_state_threshold_robustness.csv"), index=False)

    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(res["threshold_mi"], res["coef"], yerr=1.96 * res["se"], fmt="o-", capsize=4,
                markersize=6, linewidth=1.2, color=PALETTE["blue"])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(res["threshold_mi"])
    ax.set_xlabel("Border distance threshold (miles)")
    ax.set_ylabel("Border x Restricted coefficient (log points)")
    ax.set_title("Cross-regime interaction across distance thresholds\nRevenue per capita, NM+CO+MT, county-clustered 95% CI")
    fig.tight_layout()
    fig_path = os.path.join(OUT, "fig_cross_state_threshold_robustness.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}")
    return res


def continuous_exposure(df_base):
    print("\n=== 2. Continuous cross-border exposure x Restricted (75mi population) ===")
    border = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df_base.merge(border[["fips5", "nonlegal_pop_within_75mi"]], on="fips5", how="left")
    df["log_exposure"] = np.log1p(df["nonlegal_pop_within_75mi"])

    outcomes = [("log_revenue_per_capita", "Revenue per capita"),
                ("log_dispensary_per_1000", "Dispensary density"),
                ("log_revenue_per_store", "Revenue per store")]
    rows = []
    for yvar, label in outcomes:
        d = df.dropna(subset=[yvar, "log_exposure", "restricted"]).copy()
        m = smf.ols(f"{yvar} ~ log_exposure + log_exposure:restricted + C(state_period)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b_e = m.params.get("log_exposure", np.nan)
        p_e = m.pvalues.get("log_exposure", np.nan)
        b_i = m.params.get("log_exposure:restricted", np.nan)
        se_i = m.bse.get("log_exposure:restricted", np.nan)
        p_i = m.pvalues.get("log_exposure:restricted", np.nan)
        n = int(m.nobs)
        rows.append({"outcome": label, "exposure_coef": b_e, "exposure_p": p_e,
                     "interaction_coef": b_i, "interaction_se": se_i, "interaction_p": p_i, "n": n})
        print(f"{label}: exposure={b_e:+.4f} (p={p_e:.3f}), "
              f"exposure x restricted={b_i:+.4f} (se={se_i:.4f}), p={p_i:.3f}, n={n}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "cross_state_continuous_exposure_interaction.csv"), index=False)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df_base = load_base()
    threshold_robustness(df_base)
    continuous_exposure(df_base)
