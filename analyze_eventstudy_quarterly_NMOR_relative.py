"""
Quarterly border x event-time study, New Mexico + Oregon pooled, restricted
to each state's own FREE-ENTRY period only. Per explicit user request and
follow-up clarification: since OR's free period (2017Q1-2021Q4) and NM's
data window (2022Q2 onward, free its entire panel) do not overlap in
calendar time at all -- OR was already capped by the time NM's market
existed -- the two states are aligned on RELATIVE time instead: quarter 0
is each state's own first free-entry quarter in this data, quarter 1 the
next, and so on. This lets the two states' free-entry experiences be
compared/pooled even though the actual calendar dates never coincide.

Revenue per capita only: Oregon's per-store/density data is a single
current-year (2025, i.e. already-capped) snapshot repeated across periods,
so it cannot support a genuine free-period (2017-2021) event study for
those two outcomes -- using it there would apply a 2025 dispensary count to
2017-2021 revenue, which is not a real historical count. This is the same
"to the extent possible" limitation already documented for the pooled
CO+OR+MT+NM analysis.

Specification: log(revenue_per_capita) ~ border_adjacent +
border_adjacent:C(rel_q) + log_population + C(rel_q) + C(state_abbr),
county-clustered standard errors.
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


def build_relative_time(df):
    df = df.sort_values(["state_abbr", "year", "quarter"]).copy()
    df["q_index"] = df["year"] * 4 + df["quarter"]
    first_q = df.groupby("state_abbr")["q_index"].transform("min")
    df["rel_q"] = df["q_index"] - first_q
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})

    nm = df[(df.state_abbr == "NM")].copy()
    orr = df[(df.state_abbr == "OR") & (df.entry_capped == 0)].copy()
    print(f"NM (free, entire panel): {nm.yq.nunique()} quarters, {nm.yq.min()}-{nm.yq.max()}")
    print(f"OR (free period only): {orr.yq.nunique()} quarters, {orr.yq.min()}-{orr.yq.max()}")

    pooled = pd.concat([nm, orr], ignore_index=True)
    pooled = build_relative_time(pooled)
    print(f"\nPooled free-entry sample: {len(pooled)} rows, "
          f"relative quarter range 0-{pooled.rel_q.max()}")
    print(pooled.groupby("state_abbr")["rel_q"].agg(["min", "max", "nunique"]))

    d = pooled.dropna(subset=["log_revenue_per_capita", "border_adjacent", "log_population"]).copy()
    formula = ("log_revenue_per_capita ~ border_adjacent + border_adjacent:C(rel_q) + log_population "
               "+ C(rel_q) + C(state_abbr)")
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\nn={len(d)}, states={sorted(d.state_abbr.unique())}")
    print(f"Border main effect: b={m.params.get('border_adjacent', float('nan')):+.3f} "
          f"p={m.pvalues.get('border_adjacent', float('nan')):.3f}")

    all_rel_qs = sorted(d["rel_q"].unique())
    interacted = {int(t.split("[T.")[1].rstrip("]")) for t in m.params.index
                  if t.startswith("border_adjacent:C(rel_q)")}
    rows = []
    for rq in all_rel_qs:
        if rq in interacted:
            test_str = f"border_adjacent + border_adjacent:C(rel_q)[T.{rq}] = 0"
        else:
            test_str = "border_adjacent = 0"  # omitted reference relative-quarter
        tt = m.t_test(test_str)
        rows.append((rq, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
    dd = pd.DataFrame(rows, columns=["rel_q", "coef", "se"])
    dd.to_csv(os.path.join(OUT, "eventstudy_quarterly_NMOR_relative_Revenuepercapita.csv"), index=False)

    # also track which state(s) contribute at each relative quarter, for the figure/caption
    contrib = pooled.groupby("rel_q")["state_abbr"].apply(lambda s: "+".join(sorted(s.unique()))).to_dict()

    apply_healy_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xt = np.arange(len(dd))
    ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                linewidth=1.2, color=PALETTE["blue"])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    labels = [f"q{r}\n({contrib.get(r, '')})" for r in dd["rel_q"]]
    ax.set_xticks(xt)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Quarters since each state's own free-entry data begins")
    ax.set_ylabel("Border coefficient (log points)")
    ax.set_title("Border x event-time study: Revenue per capita\n"
                  "NM (free, 2022Q2+) and OR (free period, 2017Q1-2021Q4), aligned by relative quarter\n"
                  "county-clustered 95% CI", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NMOR_relative.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
