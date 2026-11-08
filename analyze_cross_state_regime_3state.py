"""
Cross-state border-premium comparison across entry regimes, THREE STATES
ONLY: New Mexico (free entry), Colorado (capped/restricted), Montana
(capped/restricted). Oregon is dropped entirely from this analysis (its own
free-to-capped regime switch made it a useful robustness check in an earlier
draft, but it is not part of the revised three-state design and its
licensing history is not discussed here).

With Oregon removed, Border_c and Restricted_s are BOTH time-invariant for
every county in this sample (CO and MT are capped throughout; NM is free
throughout) -- so a specification with county fixed effects would exactly
absorb Border_c x Restricted_s and cannot be estimated; we do not attempt it.
State x quarter fixed effects (via a concatenated state-quarter key, as
elsewhere in this project) absorb everything common to a state-quarter cell,
including the Restricted_s main effect.

Three levels of evidence, all reported (not just the pooled one), per the
revised empirical strategy:
  1. State-specific border premiums (NM, CO, MT estimated separately) --
     makes clear whether a pooled result is driven by one state.
  2. Pooled Border_c x Restricted_s interaction (CO+MT vs. NM).
  3. Pairwise comparisons: NM vs. CO only, NM vs. MT only.

All three outcomes (revenue per capita, dispensary density, revenue per
store) at all three levels. County-clustered standard errors throughout;
we do not claim this eliminates state-level confounding given only three
states are compared (documented explicitly in the paper text).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")

STATES = ["NM", "CO", "MT"]
RESTRICTED = {"NM": 0, "CO": 1, "MT": 1}
OUTCOMES = [("log_revenue_per_capita", "Revenue per capita"),
            ("log_dispensary_per_1000", "Dispensary density"),
            ("log_revenue_per_store", "Revenue per store")]


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def load(border_file="county_border_stateline.csv"):
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    n_before = len(df)
    df = df[df.state_abbr.isin(STATES)].copy()
    print(f"Dropped Oregon: {n_before} -> {len(df)} county-quarter rows "
          f"({', '.join(STATES)} only)")
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    border = pd.read_csv(os.path.join(TBL, border_file), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county"]], on="fips5", how="left")
    n_before2 = len(df)
    df = df.dropna(subset=["border_county"]).copy()
    if len(df) < n_before2:
        print(f"Dropped {n_before2 - len(df)} rows with no border classification match")
    df["restricted"] = df["state_abbr"].map(RESTRICTED)
    df["period"] = df["yq"]
    df["state_period"] = df["state_abbr"] + "_" + df["period"].astype(str)
    return df


def state_specific(df):
    print("\n=== State-specific border premiums (NM, CO, MT estimated separately) ===")
    rows = []
    for state in STATES:
        d0 = df[df.state_abbr == state]
        for yvar, label in OUTCOMES:
            d = d0.dropna(subset=[yvar, "border_county"]).copy()
            m = smf.ols(f"{yvar} ~ border_county + C(period)", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["fips5"]})
            b = m.params.get("border_county", np.nan)
            se = m.bse.get("border_county", np.nan)
            p = m.pvalues.get("border_county", np.nan)
            n = int(m.nobs)
            n_counties = d.fips5.nunique()
            n_border = d.drop_duplicates("fips5").border_county.sum()
            rows.append({"state": state, "outcome": label, "coef": b, "se": se, "p": p,
                         "n": n, "n_counties": n_counties, "n_border_counties": int(n_border)})
            print(f"{state} {label}: b={b:+.3f} ({se:.3f}), {fmt_pct(b)}, p={p:.3f}, "
                  f"n={n}, counties={n_counties} ({int(n_border)} border)")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "cross_state_border_premium_by_state.csv"), index=False)
    return res


def pooled(df, tag="full", states=None):
    d0 = df if states is None else df[df.state_abbr.isin(states)]
    label_states = "+".join(states) if states else "+".join(STATES)
    print(f"\n=== Pooled Border x Restricted interaction ({label_states}) ===")
    rows = []
    for yvar, label in OUTCOMES:
        d = d0.dropna(subset=[yvar, "border_county", "restricted"]).copy()
        m = smf.ols(f"{yvar} ~ border_county + border_county:restricted + C(state_period)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b_b = m.params.get("border_county", np.nan)
        p_b = m.pvalues.get("border_county", np.nan)
        b_i = m.params.get("border_county:restricted", np.nan)
        se_i = m.bse.get("border_county:restricted", np.nan)
        p_i = m.pvalues.get("border_county:restricted", np.nan)
        n = int(m.nobs)
        rows.append({"sample": label_states, "outcome": label, "border_coef": b_b, "border_p": p_b,
                     "interaction_coef": b_i, "interaction_se": se_i, "interaction_p": p_i, "n": n})
        print(f"{label}: border={fmt_pct(b_b)} (p={p_b:.3f}), "
              f"border x restricted={fmt_pct(b_i)} (b={b_i:+.3f}, se={se_i:.3f}, p={p_i:.3f}), n={n}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, f"cross_state_regime_interaction_{tag}.csv"), index=False)
    return res


def main():
    df = load()
    print(f"\nFinal sample: {len(df)} county-quarter rows, {df.fips5.nunique()} counties")
    print(df.groupby("state_abbr")["border_county"].agg(
        lambda s: f"{int(s.iloc[0]) if s.nunique()<=1 else 'varies'}"))

    state_res = state_specific(df)

    full_res = pooled(df, tag="full_NM_CO_MT")
    nmco_res = pooled(df, tag="NM_CO", states=["NM", "CO"])
    nmmt_res = pooled(df, tag="NM_MT", states=["NM", "MT"])

    combined = pd.concat([full_res, nmco_res, nmmt_res], ignore_index=True)
    combined.to_csv(os.path.join(TBL, "cross_state_regime_interaction_all_samples.csv"), index=False)
    print("\n=== Summary: Border x Restricted across samples ===")
    print(combined[["sample", "outcome", "interaction_coef", "interaction_p", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
