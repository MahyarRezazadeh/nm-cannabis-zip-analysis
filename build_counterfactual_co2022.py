"""
Counterfactual dispensary-revenue benchmark for Colorado border counties in
2024, under the New Mexico vs. Colorado pairwise Border x Restricted
interaction, with Colorado restricted to 2022Q2 onward (spec C -- matching
New Mexico's own panel window, so that Colorado's New-Mexico-facing border
counties are never in a period when New Mexico was still illegal).

Same construction as build_counterfactual_revenue_benchmark.py; only the
Colorado sample window changes.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
N_BOOT = 500
RNG_SEED = 12345


def load():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[df.state_abbr.isin(["NM", "CO"])].copy()
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    # spec C: Colorado restricted to 2022Q2+
    keep = (df.state_abbr != "CO") | (df.year > 2022) | ((df.year == 2022) & (df.quarter >= 2))
    df = df[keep].copy()
    b = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(b[["fips5", "border_county"]], on="fips5", how="left").dropna(subset=["border_county"])
    df["border_county"] = df["border_county"].astype(int)
    df["restricted"] = (df.state_abbr == "CO").astype(int)
    df["state_period"] = df.state_abbr + "_" + df.yq.astype(str)
    return df.dropna(subset=["log_revenue_per_capita"]).copy()


def fit_theta(d):
    m = smf.ols("log_revenue_per_capita ~ border_county + border_county:restricted + C(state_period)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return m.params.get("border_county:restricted", np.nan)


def main():
    df = load()
    theta = fit_theta(df)
    mult = np.exp(-theta)
    print(f"NM vs CO (spec C), revenue per capita: theta = {theta:+.4f}, multiplier e^-theta = {mult:.3f}")

    cap24 = df[(df.state_abbr == "CO") & (df.border_county == 1) & (df.year == 2024)].copy()
    actual = cap24["total_sales"].sum()
    cf = actual * mult
    gap = cf - actual
    print(f"CO border 2024: {cap24.fips5.nunique()} counties, {len(cap24)} county-quarters")
    print(f"Actual ${actual:,.0f}  ->  counterfactual ${cf:,.0f}  (gap ${gap:,.0f})")

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
        try:
            m = smf.ols("log_revenue_per_capita ~ border_county + border_county:restricted + C(state_period)",
                        data=bdf).fit()
            tb = m.params.get("border_county:restricted", np.nan)
        except Exception:
            continue
        if not np.isfinite(tb):
            continue
        c24 = bdf[(bdf.state_abbr == "CO") & (bdf.border_county == 1) & (bdf.year == 2024)]
        gaps.append(c24["total_sales"].sum() * (np.exp(-tb) - 1))
    gaps = np.array(gaps)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"Bootstrap reps used: {len(gaps)}/{N_BOOT}")
    print(f"95% CI on the revenue gap: (${lo:,.0f}, ${hi:,.0f})  =  ({lo/1e6:,.1f} M, {hi/1e6:,.1f} M)")

    out = pd.DataFrame([{
        "theta": theta, "multiplier": mult, "actual_revenue_2024": actual,
        "counterfactual_revenue_2024": cf, "revenue_gap": gap,
        "boot_ci_lo": lo, "boot_ci_hi": hi, "n_boot": len(gaps),
        "n_border_counties": cap24.fips5.nunique(),
    }])
    out.to_csv(os.path.join(TBL, "counterfactual_revenue_benchmark_co2022.csv"), index=False)
    print(f"\nWrote {os.path.join(TBL, 'counterfactual_revenue_benchmark_co2022.csv')}")


if __name__ == "__main__":
    main()
