"""
Stacked staggered difference-in-differences for the arrest-based cross-
border demand proxy: does the border-adjacent Oklahoma/Texas/Utah county
adjacent to a state that legalizes recreational cannabis see a rise in
marijuana possession + sale/manufacturing arrests (FBI arrest codes
158 + 153) relative to that same state's own interior counties, once its
neighbor's retail market opens?

Three staggered cohorts (a county's earliest qualifying neighbor):
  - Colorado retail 2014-01-01 -> treats bordering OK (2) + TX (3) + UT (5)
  - Arizona  retail 2021-01-22 -> treats bordering UT (4)
  - New Mexico retail 2022-04-01 -> treats bordering TX (33)

Uses the STACKED regression design (Cengiz, Dube, Lindner & Zipperer 2019;
consistent with the "bad comparisons" concern this paper's own cited
Roth, Sant'Anna, Bilinski & Poe (2023) survey raises about naive two-way-
fixed-effects staggered designs): build one clean 2-group sub-experiment
per cohort (that cohort's treated counties vs. that cohort's OWN state(s)'
never-treated interior counties, i.e. counties never within 75 miles of
ANY of the three legalizing neighbors), each with its own copy of every
county-quarter observation, then stack all three sub-experiments and
estimate a single regression with county x stack fixed effects and
stack x quarter fixed effects, so no already-treated county is ever used
as a control for a different cohort and no cross-cohort calendar-time
shock contaminates another cohort's estimate.

Specification: log(1 + arrests) ~ C(fips5) + C(stack x quarter)
  + sum_k Treated_i x 1[relative_quarter = k], county-clustered SEs.
County fixed effects use the county's own FIPS code (not a stack-specific
copy): a treated county belongs to exactly one cohort/stack by construction
(the "earliest qualifying neighbor" rule), so only never-treated control
counties ever repeat across stacks, and sharing one FE level for them across
stacks is not a contamination risk (they are never treated in any stack).
This also keeps the fixed-effect dimension (222 counties) from exceeding
the number of clustering groups, which produced degenerate (NaN) cluster-
robust standard errors under the naive county x stack FE version of this
regression. Estimated via linearmodels.PanelOLS (FE absorbed by demeaning,
not literal dummy columns), which is numerically far more stable here than
a dummy-variable OLS regression with several hundred FE columns.
"""
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")

WINDOW = 12  # relative quarters, endpoints binned

COHORTS = {
    "CO": {"date": pd.Timestamp("2014-01-01"), "states": ["OK", "TX", "UT"]},
    "AZ": {"date": pd.Timestamp("2021-01-22"), "states": ["UT"]},
    "NM": {"date": pd.Timestamp("2022-04-01"), "states": ["TX"]},
}


def month_to_period(m):
    mm, yyyy = m.split("-")
    return pd.Period(year=int(yyyy), month=int(mm), freq="Q")


def build_county_quarter_panel():
    raw = pd.read_csv(os.path.join(TBL, "arrest_pull_raw.csv"), dtype={"fips5": str, "offense_code": str})
    raw["quarter"] = raw["month"].apply(month_to_period)
    q = raw.groupby(["fips5", "quarter"])["arrests"].sum().reset_index()
    q = q.rename(columns={"arrests": "marijuana_arrests"})
    q["log_arrests"] = np.log1p(q["marijuana_arrests"])
    return q


def build_stacks(panel, sel):
    sel = sel.set_index("fips5")
    stacks = []
    for cohort, info in COHORTS.items():
        treat_q = pd.Period(info["date"], freq="Q")
        treated_fips = sel[(sel.treated_by == cohort)].index.tolist()
        control_fips = sel[(sel.border_county == 0) & (sel.state_abbr.isin(info["states"]))].index.tolist()
        stack_fips = treated_fips + control_fips
        sub = panel[panel.fips5.isin(stack_fips)].copy()
        sub["cohort"] = cohort
        sub["treated"] = sub["fips5"].isin(treated_fips).astype(int)
        sub["rel_q"] = (sub["quarter"] - treat_q).apply(lambda x: x.n if hasattr(x, "n") else int(x))
        stacks.append(sub)
        print(f"Stack {cohort}: {len(treated_fips)} treated, {len(control_fips)} control counties, "
              f"{len(sub)} county-quarter obs")
    out = pd.concat(stacks, ignore_index=True)
    out["rel_q_bin"] = out["rel_q"].clip(-WINDOW, WINDOW)
    # PanelOLS requires a numeric/date-like time index; stack_id*100000 + quarter's
    # ordinal keeps each stack's quarters numeric, monotonic, and non-overlapping
    # with the other stacks' quarter ranges.
    stack_num = {c: i for i, c in enumerate(COHORTS)}
    out["stack_quarter"] = out["cohort"].map(stack_num) * 100000 + out["quarter"].apply(lambda p: p.ordinal)
    return out


def bin_label(b):
    return f"m{abs(b)}" if b < 0 else f"p{b}"


def to_panel(d, extra_cols):
    p = d.set_index(["fips5", "stack_quarter"])[extra_cols + ["log_arrests"]].copy()
    return p


def run_event_study(stacked):
    d = stacked.copy()
    bins = sorted(d["rel_q_bin"].unique())
    ref = -1
    dummy_cols = []
    for b in bins:
        if b == ref:
            continue
        col = f"rq_{bin_label(b)}"
        d[col] = ((d["rel_q_bin"] == b) & (d["treated"] == 1)).astype(float)
        dummy_cols.append(col)

    panel = to_panel(d, dummy_cols)
    y = panel["log_arrests"]
    X = panel[dummy_cols]
    m = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)

    rows = []
    for b in bins:
        if b == ref:
            rows.append((b, 0.0, 0.0, np.nan))
            continue
        key = f"rq_{bin_label(b)}"
        if key not in m.params.index:
            continue
        rows.append((b, m.params[key], m.std_errors[key], m.pvalues[key]))
    ev = pd.DataFrame(rows, columns=["rel_q", "coef", "se", "p"])
    return ev, m


def run_static_did(stacked):
    d = stacked.copy()
    d["post"] = (d["rel_q"] >= 0).astype(int)
    d["treated_post"] = (d["treated"] * d["post"]).astype(float)
    panel = to_panel(d, ["treated_post"])
    y = panel["log_arrests"]
    X = panel[["treated_post"]]
    m = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)
    return m


def main():
    panel = build_county_quarter_panel()
    sel = pd.read_csv(os.path.join(TBL, "arrest_county_agency_selection.csv"), dtype={"fips5": str})

    stacked = build_stacks(panel, sel)
    stacked.to_csv(os.path.join(TBL, "arrest_stacked_panel.csv"), index=False)

    print("\n=== Pooled stacked event study (all 3 cohorts) ===")
    ev, m = run_event_study(stacked)
    print(ev.to_string(index=False))
    print(f"n={int(m.nobs)}, R2={m.rsquared:.3f}")

    print("\n=== Pooled stacked static DID ===")
    m_static = run_static_did(stacked)
    print(f"treated_post: {m_static.params['treated_post']:+.3f} "
          f"({m_static.std_errors['treated_post']:.3f}), p={m_static.pvalues['treated_post']:.3f}, "
          f"n={int(m_static.nobs)}")

    print("\n=== Per-cohort static DID ===")
    per_cohort = {}
    for cohort in COHORTS:
        sub = stacked[stacked["cohort"] == cohort]
        mc = run_static_did(sub)
        per_cohort[cohort] = mc
        print(f"{cohort}: treated_post={mc.params['treated_post']:+.3f} "
              f"({mc.std_errors['treated_post']:.3f}), p={mc.pvalues['treated_post']:.3f}, n={int(mc.nobs)}")

    ev.to_csv(os.path.join(OUT, "arrest_eventstudy_coefs.csv"), index=False)

    print("\n=== Per-cohort event studies ===")
    cohort_ev = {}
    for cohort in COHORTS:
        sub = stacked[stacked["cohort"] == cohort]
        ev_c, m_c = run_event_study(sub)
        cohort_ev[cohort] = ev_c
        print(f"\n--- {cohort} ---")
        print(ev_c.to_string(index=False))

    # ---------- figures ----------
    apply_healy_style()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(ev["rel_q"], ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=3,
                markersize=4, linewidth=1.2, color=PALETTE["blue"])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Quarters relative to neighboring state's legalization")
    ax.set_ylabel("Coefficient on log(1+marijuana arrests)")
    ax.set_title("Stacked event study: marijuana arrests in border counties of OK/TX/UT\n"
                  "around a neighboring state's legalization, county-clustered 95% CI")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_arrest_eventstudy.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote fig_arrest_eventstudy.png")

    cohort_labels = {"CO": "Colorado neighbors (OK, TX, UT border counties)",
                      "AZ": "Arizona neighbors (UT border counties)",
                      "NM": "New Mexico neighbors (TX border counties)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        ev_c = cohort_ev[cohort]
        ax.errorbar(ev_c["rel_q"], ev_c["coef"], yerr=1.96 * ev_c["se"], fmt="o-", capsize=3,
                     markersize=4, linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(-0.5, color="red", linewidth=0.8, linestyle=":")
        ax.set_title(cohort_labels[cohort], fontsize=10)
        ax.set_xlabel("Quarters relative to legalization")
    axes[0].set_ylabel("Coefficient on log(1+marijuana arrests)")
    fig.suptitle("Per-cohort event studies: marijuana arrests in border counties around each\n"
                 "neighboring state's legalization, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_arrest_eventstudy_percohort.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig_arrest_eventstudy_percohort.png")


if __name__ == "__main__":
    main()
