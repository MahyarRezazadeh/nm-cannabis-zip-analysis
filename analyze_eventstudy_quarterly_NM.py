"""
Quarterly border x period event study, New Mexico only, per explicit user
request. Uses master_quarterly_panel_no_wa.csv (already NM's genuine
quarterly panel -- real dispensary counts, not a snapshot, so all three
outcomes are estimable here, unlike the pooled CO+NM check).

Specification: log(y) ~ border_adjacent + border_adjacent:C(yq) +
log_population + C(yq) + C(state_abbr), county-clustered standard errors.
C(state_abbr) is included per explicit instruction even though NM-only
means it contributes nothing (a single-level categorical is absorbed into
the intercept by patsy's treatment coding, so it appears in the formula but
adds no columns/estimates) -- kept for consistency with the pooled spec.
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


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[df.state_abbr == "NM"].copy()
    df["period"] = df["yq"]
    print(f"NM quarterly panel: {len(df)} rows, {df.fips5.nunique()} counties, "
          f"periods {df.period.min()}-{df.period.max()}")

    specs = [
        ("Revenue per capita", "log_revenue_per_capita"),
        ("Revenue per store", "log_revenue_per_store"),
        ("Dispensary density", "log_dispensary_per_1000"),
    ]

    es_results = {}
    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_adjacent", "log_population"]).copy()
        formula = (f"{yvar} ~ border_adjacent + border_adjacent:C(period) + log_population "
                   f"+ C(period) + C(state_abbr)")
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

        all_periods = sorted(d["period"].unique())
        interacted = {t.split("[T.")[1].rstrip("]") for t in m.params.index
                      if t.startswith("border_adjacent:C(period)")}
        rows = []
        for p in all_periods:
            if p in interacted:
                test_str = f"border_adjacent + border_adjacent:C(period)[T.{p}] = 0"
            else:
                test_str = "border_adjacent = 0"  # omitted reference period
            tt = m.t_test(test_str)
            rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["period", "coef", "se"])
        es_results[label] = dd
        dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_{label.replace(' ', '')}.csv"), index=False)
        print(f"{label}: n={len(d)}, {len(all_periods)} quarters, "
              f"border main effect b={m.params.get('border_adjacent', float('nan')):+.3f} "
              f"p={m.pvalues.get('border_adjacent', float('nan')):.3f}")

    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (label, dd) in zip(axes, es_results.items()):
        xt = np.arange(len(dd))
        ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                    linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(dd["period"], rotation=90, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Border coefficient (log points)")
    fig.suptitle("Border x quarter event study -- New Mexico only, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
