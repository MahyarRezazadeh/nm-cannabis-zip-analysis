"""
Sensitivity of the cross-state entry-regime estimation to how Colorado's
New-Mexico-facing border counties are treated before April 2022, when New
Mexico still had no legal recreational market.

Five Colorado counties border ONLY New Mexico -- La Plata (Durango),
Las Animas (Trinidad), Costilla, Conejos, Archuleta. The paper's fixed
border indicator (post-2022 snapshot, New Mexico legal) codes them interior
for the whole 2015--2025 panel, even though 2015--2021 they faced a captive
illegal New Mexico market.

Three specifications, otherwise identical to analyze_cross_state_regime_3state.py:

  (A) Baseline      -- fixed border_county (county_border_stateline.csv).
  (B) Time-varying  -- the five NM-only-border Colorado counties are border
                       for year < 2022 and interior from 2022 on; everyone
                       else unchanged (no other neighbor changes status
                       within any sample state's window).
  (C) CO 2022+      -- fixed border_county, but Colorado is restricted to
                       2022Q2 onward, matching New Mexico's own window.

Reports the pooled 3-state and the NM-vs-CO pairwise Border x Restricted
interaction for all three outcomes, plus the state-specific Colorado border
premium and the implied revenue counterfactual.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
EG = os.path.join(os.path.dirname(BASE), "entry game-marijuana")

STATES = ["NM", "CO", "MT"]
RESTRICTED = {"NM": 0, "CO": 1, "MT": 1}
OUTCOMES = [("log_revenue_per_capita", "Revenue per capita"),
            ("log_dispensary_per_1000", "Dispensary density"),
            ("log_revenue_per_store", "Revenue per store")]
CO_NM_ONLY = ["08067", "08071", "08023", "08021", "08007"]  # La Plata, Las Animas, Costilla, Conejos, Archuleta


def _haversine_mi(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def co_timevarying_border():
    """
    Recompute Colorado's centroid border rule PER YEAR: a CO county is border
    in year t if its centroid is within 75 mi of the line of a state that is
    nonlegal in t, with >=1,000 residents of that nonlegal state within 75 mi.
    New Mexico is nonlegal for t < 2022 and legal from 2022; every other CO
    neighbor (UT/WY/NE/KS/OK) is nonlegal throughout.
    Returns dict {(fips5, year) -> 1/0} for CO counties.
    """
    import geopandas as gpd
    c = gpd.read_file(os.path.join(EG, "data", "us_counties_shp", "cb_2023_us_county_500k.shp"))
    c = c[~c["STUSPS"].isin({"AK", "HI", "PR", "VI", "GU", "AS", "MP"})].to_crs(5070)
    st = c.dissolve(by="STUSPS").reset_index()[["STUSPS", "geometry"]]
    co = c[c["STUSPS"] == "CO"].copy()
    co["cent"] = co.geometry.centroid

    zips = pd.read_csv(os.path.join(EG, "data", "raw", "uszips_simplemaps.csv"),
                       dtype={"zip": str}).dropna(subset=["lat", "lng", "population"])
    always_nonlegal = ["UT", "WY", "NE", "KS", "OK"]

    line_fixed = st[st.STUSPS.isin(always_nonlegal)].union_all().boundary
    line_with_nm = st[st.STUSPS.isin(always_nonlegal + ["NM"])].union_all().boundary
    d_fixed = co["cent"].distance(line_fixed) / 1609.34
    d_withnm = co["cent"].distance(line_with_nm) / 1609.34

    zc = co.to_crs(4326)
    cent_ll = zc.geometry.centroid
    lat_c, lon_c = cent_ll.y.to_numpy(), cent_ll.x.to_numpy()

    def pop_within(states):
        z = zips[zips["state_id"].isin(states)]
        zl, zo, zp = z["lat"].to_numpy(), z["lng"].to_numpy(), z["population"].to_numpy()
        return np.array([zp[_haversine_mi(la, lo, zl, zo) <= 75].sum() for la, lo in zip(lat_c, lon_c)])

    pop_fixed = pop_within(always_nonlegal)
    pop_withnm = pop_within(always_nonlegal + ["NM"])

    out = {}
    for i, r in enumerate(co.itertuples()):
        for yr in range(2015, 2026):
            if yr < 2022:
                b = int((d_withnm.iloc[i] <= 75) and (pop_withnm[i] >= 1000))
            else:
                b = int((d_fixed.iloc[i] <= 75) and (pop_fixed[i] >= 1000))
            out[(r.GEOID, yr)] = b
    return out


def base_panel():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[df.state_abbr.isin(STATES)].copy()
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    b = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(b[["fips5", "border_county"]], on="fips5", how="left").dropna(subset=["border_county"])
    df["border_county"] = df["border_county"].astype(int)
    df["restricted"] = df["state_abbr"].map(RESTRICTED)
    df["state_period"] = df["state_abbr"] + "_" + df["yq"].astype(str)
    df["period"] = df["yq"]
    return df


_TV_CACHE = {}


def make_spec(spec):
    df = base_panel()
    if spec == "A":
        pass
    elif spec == "B":
        pre = df["fips5"].isin(CO_NM_ONLY) & (df["year"] < 2022)
        df.loc[pre, "border_county"] = 1
    elif spec == "B2":
        if "tv" not in _TV_CACHE:
            _TV_CACHE["tv"] = co_timevarying_border()
        tv = _TV_CACHE["tv"]
        co_mask = df["state_abbr"] == "CO"
        df.loc[co_mask, "border_county"] = [
            tv.get((f, int(y)), b) for f, y, b in
            zip(df.loc[co_mask, "fips5"], df.loc[co_mask, "year"], df.loc[co_mask, "border_county"])
        ]
        df["border_county"] = df["border_county"].astype(int)
    elif spec == "C":
        keep = (df.state_abbr != "CO") | (df.year > 2022) | ((df.year == 2022) & (df.quarter >= 2))
        df = df[keep].copy()
    return df


def pooled_interaction(df, states, yvar):
    d = df[df.state_abbr.isin(states)].dropna(subset=[yvar, "border_county", "restricted"]).copy()
    m = smf.ols(f"{yvar} ~ border_county + border_county:restricted + C(state_period)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return dict(theta=m.params.get("border_county:restricted", np.nan),
               se=m.bse.get("border_county:restricted", np.nan),
               p=m.pvalues.get("border_county:restricted", np.nan),
               border=m.params.get("border_county", np.nan),
               n=int(m.nobs))


def state_border_premium(df, state, yvar):
    d = df[df.state_abbr == state].dropna(subset=[yvar, "border_county"]).copy()
    m = smf.ols(f"{yvar} ~ border_county + C(period)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return dict(coef=m.params.get("border_county", np.nan),
               p=m.pvalues.get("border_county", np.nan),
               n=int(m.nobs),
               n_border=int(d.drop_duplicates("fips5").border_county.sum()))


def counterfactual(df, theta):
    """CO border-county 2024 revenue x e^{-theta}; gap vs actual."""
    b = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    d = df.merge(b[["fips5"]].assign(_bc=1), on="fips5", how="left")  # placeholder, use df border
    co24 = df[(df.state_abbr == "CO") & (df.border_county == 1) & (df.year == 2024)]
    actual = co24["total_sales"].sum()
    mult = np.exp(-theta)
    return actual, actual * mult, actual * (mult - 1)


def main():
    labels = {"A": "(A) Baseline (fixed border)",
              "B": "(B) Time-varying, 5 CO-NM-only counties",
              "B2": "(B2) Time-varying, full CO centroid recompute",
              "C": "(C) Colorado restricted to 2022+"}
    rows = []
    for spec in ["A", "B", "B2", "C"]:
        df = make_spec(spec)
        n_co_b = df[df.state_abbr == "CO"].drop_duplicates("fips5").border_county.sum()
        n_co_bcq = int(((df.state_abbr == "CO") & (df.border_county == 1)).sum())
        print(f"\n########## {labels[spec]} ##########")
        print(f"CO county-quarters: {int((df.state_abbr=='CO').sum())}  "
              f"(border cq: {n_co_bcq}); border CO counties ever: {int(n_co_b)}")
        for yvar, oname in OUTCOMES:
            pool3 = pooled_interaction(df, STATES, yvar)
            nmco = pooled_interaction(df, ["NM", "CO"], yvar)
            cosp = state_border_premium(df, "CO", yvar)
            print(f"  {oname}:")
            print(f"     pooled 3-state  theta={pool3['theta']:+.3f} (se {pool3['se']:.3f}, p={pool3['p']:.3f}), n={pool3['n']}")
            print(f"     NM vs CO        theta={nmco['theta']:+.3f} (se {nmco['se']:.3f}, p={nmco['p']:.3f}), n={nmco['n']}")
            print(f"     CO border premium alone: {(np.exp(cosp['coef'])-1)*100:+.1f}% (p={cosp['p']:.3f}), n={cosp['n']}")
            rows.append(dict(spec=labels[spec], outcome=oname,
                             pooled_theta=pool3["theta"], pooled_p=pool3["p"], pooled_n=pool3["n"],
                             nmco_theta=nmco["theta"], nmco_p=nmco["p"], nmco_n=nmco["n"],
                             co_premium_pct=(np.exp(cosp["coef"]) - 1) * 100, co_premium_p=cosp["p"]))
            if oname == "Revenue per capita":
                act, cf, gap = counterfactual(df, nmco["theta"])
                print(f"     [rev/capita] NM-CO theta multiplier e^-theta={np.exp(-nmco['theta']):.3f}; "
                      f"CO border 2024 actual ${act/1e6:,.1f}M -> counterfactual ${cf/1e6:,.1f}M "
                      f"(gap ${gap/1e6:,.1f}M)")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "cross_state_regime_timevarying_comparison.csv"), index=False)
    print(f"\nWrote {os.path.join(TBL, 'cross_state_regime_timevarying_comparison.csv')}")

    # compact comparison table
    print("\n\n================ COMPARISON: Border x Restricted interaction (theta) ================")
    for oname, _ in [(o[1], o[0]) for o in OUTCOMES]:
        print(f"\n--- {oname} ---")
        print(f"{'Specification':<38} {'pooled 3-state theta (p)':<28} {'NM-CO theta (p)':<24} {'CO premium alone':<18}")
        for lab in labels.values():
            r = res[(res.spec == lab) & (res.outcome == oname)].iloc[0]
            print(f"{lab:<38} {r.pooled_theta:+.3f} ({r.pooled_p:.3f})".ljust(66) +
                  f"{r.nmco_theta:+.3f} ({r.nmco_p:.3f})".ljust(24) +
                  f"{r.co_premium_pct:+.1f}% ({r.co_premium_p:.2f})")


if __name__ == "__main__":
    main()
