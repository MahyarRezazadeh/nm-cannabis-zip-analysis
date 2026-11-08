"""
Counterfactual dispensary-revenue benchmark for Colorado border counties in 2024,
based on the DISPENSARY-DENSITY interaction (the outcome for which the
Border x Restricted interaction is statistically significant), spec C
(Colorado restricted to 2022Q2 onward).

Logic: the border premium in New Mexico operates on the extensive margin -- more
dispensaries. theta_density is the amount by which Colorado's density border
premium falls short of New Mexico's. Applying the free-entry benchmark
multiplier e^{-theta_density} to each Colorado border county's own 2024
dispensary revenue asks how much revenue those counties would have generated if
their border markets had expanded (in number of stores) the way New Mexico's do,
holding revenue per store at its observed Colorado level.

Parallel to build_counterfactual_co2022.py; only the outcome used to estimate
theta changes (log_dispensary_per_1000 instead of log_revenue_per_capita).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
N_BOOT = 500
RNG_SEED = 12345
OUTCOME = "log_dispensary_per_1000"


def load():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[df.state_abbr.isin(["NM", "CO"])].copy()
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    keep = (df.state_abbr != "CO") | (df.year > 2022) | ((df.year == 2022) & (df.quarter >= 2))
    df = df[keep].copy()
    b = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(b[["fips5", "border_county"]], on="fips5", how="left").dropna(subset=["border_county"])
    df["border_county"] = df["border_county"].astype(int)
    df["restricted"] = (df.state_abbr == "CO").astype(int)
    df["state_period"] = df.state_abbr + "_" + df.yq.astype(str)
    return df


def fit_theta(d, outcome=OUTCOME):
    d = d.dropna(subset=[outcome]).copy()
    m = smf.ols(f"{outcome} ~ border_county + border_county:restricted + C(state_period)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return (m.params.get("border_county:restricted", np.nan),
            m.bse.get("border_county:restricted", np.nan),
            m.pvalues.get("border_county:restricted", np.nan),
            len(d), d["fips5"].nunique())


def main():
    df = load()
    theta, se, p, n, ncl = fit_theta(df)
    mult = np.exp(-theta)
    print(f"NM vs CO (spec C), {OUTCOME}: theta = {theta:+.4f} (se {se:.3f}, p {p:.3f}), "
          f"n = {n} county-quarters, {ncl} counties")
    print(f"free-entry benchmark multiplier e^-theta = {mult:.3f}")

    # revenue-per-capita theta, for the table row that stays as reference
    tt = fit_theta(df, "log_revenue_per_capita")
    print(f"(revenue per capita theta = {tt[0]:+.4f}, se {tt[1]:.3f}, p {tt[2]:.3f})")

    cap24 = df[(df.state_abbr == "CO") & (df.border_county == 1) & (df.year == 2024)].copy()
    actual = cap24["total_sales"].sum()
    cf = actual * mult
    gap = cf - actual
    print(f"\nCO border 2024: {cap24.fips5.nunique()} counties, {len(cap24)} county-quarters")
    print(f"Actual ${actual:,.0f}  ->  counterfactual ${cf:,.0f}  (gap ${gap:,.0f})")
    print(f"  = actual ${actual/1e6:,.1f} M  ->  ${cf/1e6:,.1f} M  (gap +${gap/1e6:,.1f} M)")

    rng = np.random.default_rng(RNG_SEED)
    counties = df[["fips5", "state_abbr"]].drop_duplicates()
    gaps = []
    for _ in range(N_BOOT):
        rs = []
        for st in ["NM", "CO"]:
            pool = counties.loc[counties.state_abbr == st, "fips5"].to_numpy()
            rs.append(rng.choice(pool, size=len(pool), replace=True))
        rs = np.concatenate(rs)
        bdf = pd.concat([df[df.fips5 == f] for f in rs], ignore_index=True)
        bfit = bdf.dropna(subset=[OUTCOME])
        try:
            m = smf.ols(f"{OUTCOME} ~ border_county + border_county:restricted + C(state_period)",
                        data=bfit).fit()
            tb = m.params.get("border_county:restricted", np.nan)
        except Exception:
            continue
        if not np.isfinite(tb):
            continue
        c24 = bdf[(bdf.state_abbr == "CO") & (bdf.border_county == 1) & (bdf.year == 2024)]
        gaps.append(c24["total_sales"].sum() * (np.exp(-tb) - 1))
    gaps = np.array(gaps)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"\nBootstrap reps used: {len(gaps)}/{N_BOOT}")
    print(f"95% CI on the revenue gap: (${lo:,.0f}, ${hi:,.0f})  =  ({lo/1e6:,.1f} M, {hi/1e6:,.1f} M)")

    out = pd.DataFrame([{
        "outcome": OUTCOME, "theta": theta, "theta_se": se, "theta_p": p,
        "multiplier": mult, "actual_revenue_2024": actual,
        "counterfactual_revenue_2024": cf, "revenue_gap": gap,
        "boot_ci_lo": lo, "boot_ci_hi": hi, "n_boot": len(gaps),
        "n_border_counties": cap24.fips5.nunique(),
    }])
    out.to_csv(os.path.join(TBL, "counterfactual_revenue_benchmark_co2022_density.csv"), index=False)
    print(f"\nWrote {os.path.join(TBL, 'counterfactual_revenue_benchmark_co2022_density.csv')}")


if __name__ == "__main__":
    main()
