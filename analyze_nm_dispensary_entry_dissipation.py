"""
Does the New Mexico dispensary-level border premium null (see
analyze_nm_dispensary_zipFE_dynamic.py) reflect genuine competitive
dissipation -- an initial per-store premium that gets competed away as more
dispensaries enter the same ZIP -- or a from-the-start density-only effect
with no per-store margin ever? Both stories predict the SAME flat, pooled
null already found; only a design that isolates local-competition intensity
can tell them apart.

Border status: current predetermined ZIP-level classification
(tables/nm_zip_border_stateline.csv), merged via each dispensary's own
zip5_nearest -- same as analyze_nm_dispensary_zipFE_dynamic.py.

Three specs, all log_revenue ~ ... + log_zip_population + C(yq),
ZIP-clustered SEs (no ZIP FE -- needed here since Border's main effect and
its interactions are the object of interest, not just the dynamic path):

  1. Border x is_sole_provider (contemporaneous): is_sole_provider=1 if this
     dispensary is the ONLY one active in its ZIP-quarter (n_dispensaries_zip
     == 1). Tests whether the border premium concentrates in periods of local
     monopoly, pooling across all dispensaries/ZIPs.
  2. Border x log(n_dispensaries_zip): continuous version -- does the border
     coefficient shrink monotonically as local density rises?
  3. Restricted to FIRST-MOVER dispensaries only (rank_in_zip == 1, the
     earliest entrant in each ZIP by entry_year/entry_month): Border x
     is_sole_provider, now isolating literally the same firm's own revenue
     before vs. after a second dispensary shows up in its ZIP -- the
     cleanest test of "did the pioneer's own premium fade once joined by
     competitors," free of any first-mover-vs-later-entrant compositional
     difference that spec 1 cannot rule out.

Because log_zip_population enters as a free covariate in every spec (no ZIP
FE here), log_revenue and log_revenue_per_resident give IDENTICAL
coefficients on every Border-related term (see prior script's notes) --
only log_revenue is run, per that already-established identity.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def fmt_pct(b):
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def build():
    disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"), dtype={"zip5_nearest": str})
    border = pd.read_csv(os.path.join(TBL, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
    disp = disp.merge(
        border[["zip5", "border_zip"]].rename(columns={"zip5": "zip5_nearest"}),
        on="zip5_nearest", how="left")
    disp = disp.dropna(subset=["border_zip", "zip_population"]).copy()
    disp["border_zip"] = disp["border_zip"].astype(int)
    disp = disp[disp["total_sales"] > 0].copy()

    disp["log_zip_population"] = np.log(disp["zip_population"])
    disp["log_n_dispensaries_zip"] = np.log(disp["n_dispensaries_zip"])
    disp["is_sole_provider"] = (disp["n_dispensaries_zip"] == 1).astype(int)

    # rank_in_zip: order of entry among dispensaries sharing a ZIP, by entry
    # (year, month), tie-broken by dispensary_id for full determinism.
    first_seen = disp.drop_duplicates("dispensary_id")[
        ["dispensary_id", "zip5_nearest", "entry_year", "entry_month"]].copy()
    first_seen = first_seen.sort_values(["zip5_nearest", "entry_year", "entry_month", "dispensary_id"])
    first_seen["rank_in_zip"] = first_seen.groupby("zip5_nearest").cumcount() + 1
    disp = disp.merge(first_seen[["dispensary_id", "rank_in_zip"]], on="dispensary_id", how="left")
    disp["is_first_mover"] = (disp["rank_in_zip"] == 1).astype(int)

    n_zips_multi = disp.groupby("zip5_nearest")["dispensary_id"].nunique()
    print(f"ZIPs with >1 dispensary ever: {(n_zips_multi > 1).sum()} of {n_zips_multi.shape[0]}")
    print(f"First-mover dispensary-quarter rows: {disp.is_first_mover.sum()} "
          f"({disp.loc[disp.is_first_mover==1,'dispensary_id'].nunique()} distinct dispensaries)")
    print(f"Sole-provider dispensary-quarter rows: {disp.is_sole_provider.sum()} of {len(disp)}")
    return disp


def run_spec(d, formula, label, groups_col="zip5_nearest"):
    d = d.dropna(subset=["log_revenue", "log_zip_population", "border_zip"]).copy()
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[groups_col]})
    print(f"\n--- {label} ---")
    print(f"n={int(m.nobs)}, dispensaries={d.dispensary_id.nunique()}, ZIPs={d.zip5_nearest.nunique()}")
    for term in m.params.index:
        if term in ("Intercept",) or term.startswith("C(yq)"):
            continue
        b, se, p = m.params[term], m.bse[term], m.pvalues[term]
        print(f"  {term}: b={b:+.3f} ({se:.3f}), {fmt_pct(b)}, p={p:.3f}")
    return m


def main():
    d = build()

    print("\n" + "=" * 70)
    print("SPEC 1: Border x is_sole_provider (contemporaneous), ALL dispensaries")
    print("=" * 70)
    m1 = run_spec(d, "log_revenue ~ border_zip * is_sole_provider + log_zip_population + C(yq)",
                  "Spec 1: all dispensaries")

    print("\n" + "=" * 70)
    print("SPEC 2: Border x log(n_dispensaries_zip), continuous, ALL dispensaries")
    print("=" * 70)
    m2 = run_spec(d, "log_revenue ~ border_zip * log_n_dispensaries_zip + log_zip_population + C(yq)",
                  "Spec 2: all dispensaries")

    print("\n" + "=" * 70)
    print("SPEC 3: Border x is_sole_provider, FIRST-MOVERS ONLY")
    print("(isolates the same firm's own revenue before vs. after a 2nd entrant)")
    print("=" * 70)
    d_fm = d[d.is_first_mover == 1].copy()
    m3 = run_spec(d_fm, "log_revenue ~ border_zip * is_sole_provider + log_zip_population + C(yq)",
                  "Spec 3: first-movers only", groups_col="zip5_nearest")

    print("\n" + "=" * 70)
    print("Descriptive check: how long does a first-mover stay a sole provider, by border status?")
    print("=" * 70)
    fm = d[d.is_first_mover == 1].copy()
    tenure = fm.groupby(["dispensary_id", "border_zip"])["is_sole_provider"].agg(
        quarters_sole="sum", quarters_total="count").reset_index()
    print(tenure.groupby("border_zip")[["quarters_sole", "quarters_total"]].mean())

    rows = []
    for name, m, spec_n in [("Sole provider (all)", m1, 1), ("Log density (all)", m2, 2),
                             ("Sole provider (first-movers only)", m3, 3)]:
        for term in m.params.index:
            if term in ("Intercept",) or term.startswith("C(yq)"):
                continue
            rows.append({"spec": name, "spec_n": spec_n, "term": term,
                         "coef": m.params[term], "se": m.bse[term], "p": m.pvalues[term],
                         "n": int(m.nobs)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TBL, "nm_dispensary_entry_dissipation_results.csv"), index=False)
    print(f"\nWrote {os.path.join(TBL, 'nm_dispensary_entry_dissipation_results.csv')}")


if __name__ == "__main__":
    main()
