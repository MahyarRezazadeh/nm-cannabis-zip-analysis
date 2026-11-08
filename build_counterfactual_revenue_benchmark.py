"""
Model-implied dispensary-revenue benchmark: how much additional dispensary
SALES revenue (not government tax revenue -- this project has no tax-rate
data merged into this specification) would Colorado and Montana's border
counties generate if their border premium matched New Mexico's free-entry
border premium, instead of their own actual (restricted-entry) border
premium?

This is a model-based counterfactual conditional on the cross-state
Border x Restricted specification (analyze_cross_state_regime_3state.py,
NM+CO+MT pooled, 75-mile definition, theta = interaction coefficient), NOT
a clean experimentally identified fiscal-loss estimate: entry regime is not
randomly assigned across these three states, so theta is a difference in
estimated border premia across regimes, not an unqualified causal effect
(see the identification discussion in the paper text).

Method: for every CO/MT border county-quarter in 2024 (the same reference
year used in the original draft), take the county's actual observed
dispensary revenue and multiply by exp(-theta) to get its free-entry-
benchmark counterfactual revenue -- i.e., apply the estimated gap at the
county-quarter level and sum, rather than applying a single multiplier to
one pre-aggregated total (the two are algebraically identical under this
log-linear, constant-theta specification with no county-varying slopes, but
computing it county-by-county keeps the calculation transparent and is what
a bootstrap over counties needs).

Uncertainty: a county-cluster bootstrap (resample counties with replacement
within each state, refit the full NM+CO+MT interaction regression, recompute
the aggregate counterfactual gap) gives a 95% interval, reported alongside
the point estimate.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")

STATES = ["NM", "CO", "MT"]
RESTRICTED = {"NM": 0, "CO": 1, "MT": 1}
N_BOOT = 500
RNG_SEED = 12345


def load():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[df.state_abbr.isin(STATES)].copy()
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    border = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county"]], on="fips5", how="left")
    df["restricted"] = df["state_abbr"].map(RESTRICTED)
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    return df.dropna(subset=["log_revenue_per_capita", "border_county", "restricted"]).copy()


def fit_theta(d):
    m = smf.ols("log_revenue_per_capita ~ border_county + border_county:restricted + C(state_period)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return m.params.get("border_county:restricted", np.nan)


def main():
    df = load()
    theta = fit_theta(df)
    print(f"Point estimate: theta (Border x Restricted, NM+CO+MT, 75mi) = {theta:+.4f}")
    print(f"Implied: capped-regime border premium is {100*(1-np.exp(theta)):.1f}% smaller "
          f"than the free-entry (NM) border premium")
    mult = np.exp(-theta)
    print(f"Free-entry benchmark multiplier applied to actual restricted-regime revenue: {mult:.3f}x")

    # ---------- 2024 border counties in CO/MT ----------
    cap2024 = df[(df.restricted == 1) & (df.border_county == 1) & (df.year == 2024)].copy()
    cap2024["counterfactual_revenue"] = cap2024["total_sales"] * mult
    actual_total = cap2024["total_sales"].sum()
    counterfactual_total = cap2024["counterfactual_revenue"].sum()
    gap = counterfactual_total - actual_total

    print(f"\n2024 CO+MT border-county-quarters: {len(cap2024)}, "
          f"{cap2024.fips5.nunique()} distinct counties")
    for state in ["CO", "MT"]:
        sub = cap2024[cap2024.state_abbr == state]
        print(f"  {state}: {sub.fips5.nunique()} border counties, "
              f"actual ${sub.total_sales.sum():,.0f}, "
              f"counterfactual ${sub.counterfactual_revenue.sum():,.0f}")
    print(f"\nActual 2024 CO+MT border-county dispensary revenue:        ${actual_total:,.0f}")
    print(f"Free-entry-benchmark counterfactual revenue:                ${counterfactual_total:,.0f}")
    print(f"Model-implied additional dispensary revenue gap:            ${gap:,.0f}")

    # ---------- county-cluster bootstrap for uncertainty ----------
    print(f"\nRunning {N_BOOT}-replicate county-cluster bootstrap...")
    rng = np.random.default_rng(RNG_SEED)
    counties = df[["fips5", "state_abbr"]].drop_duplicates()
    boot_gaps = []
    for b in range(N_BOOT):
        resampled_fips = []
        for state in STATES:
            pool = counties.loc[counties.state_abbr == state, "fips5"].to_numpy()
            resampled_fips.append(rng.choice(pool, size=len(pool), replace=True))
        resampled_fips = np.concatenate(resampled_fips)
        boot_df = pd.concat([df[df.fips5 == f] for f in resampled_fips], ignore_index=True)
        try:
            m = smf.ols("log_revenue_per_capita ~ border_county + border_county:restricted + C(state_period)",
                        data=boot_df).fit()
            theta_b = m.params.get("border_county:restricted", np.nan)
        except Exception:
            continue
        if not np.isfinite(theta_b):
            continue
        mult_b = np.exp(-theta_b)
        cap2024_b = boot_df[(boot_df.restricted == 1) & (boot_df.border_county == 1) & (boot_df.year == 2024)]
        gap_b = cap2024_b["total_sales"].sum() * (mult_b - 1)
        boot_gaps.append(gap_b)
    boot_gaps = np.array(boot_gaps)
    lo, hi = np.percentile(boot_gaps, [2.5, 97.5])
    print(f"Bootstrap replicates used: {len(boot_gaps)} of {N_BOOT}")
    print(f"95% bootstrap interval for the revenue gap: (${lo:,.0f}, ${hi:,.0f})")

    out = pd.DataFrame([{
        "theta": theta, "multiplier": mult, "actual_revenue_2024": actual_total,
        "counterfactual_revenue_2024": counterfactual_total, "revenue_gap": gap,
        "boot_ci_lo": lo, "boot_ci_hi": hi, "n_boot": len(boot_gaps),
    }])
    out_path = os.path.join(TBL, "counterfactual_revenue_benchmark.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
