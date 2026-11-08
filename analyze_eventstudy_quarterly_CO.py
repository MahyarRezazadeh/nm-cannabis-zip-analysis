"""
Quarterly border x period event study, Colorado only, per explicit user
request. Revenue per capita only: CO's own DOR file only carries a real
monthly/quarterly retailer count starting 2025Q1 (verified directly against
the raw source -- every prior quarter back to 2015 has that field blank),
so revenue per store and dispensary density would have just 4 usable
quarters here -- not enough for a meaningful event-study line, so they are
skipped rather than plotted misleadingly. Revenue per capita has no such
limitation and is estimated over CO's full quarterly panel, 2015Q1-2025Q4.

Specification: log(revenue_per_capita) ~ border_adjacent +
border_adjacent:C(yq) + log_population + C(yq) + C(state_abbr),
county-clustered standard errors. C(state_abbr) is included per the same
instruction as the NM-only script even though CO-only means it contributes
nothing (absorbed into the intercept).
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
    df = df[df.state_abbr == "CO"].copy()
    df["period"] = df["yq"]

    n_disp_quarters = df[df.count_is_snapshot == 0].dropna(subset=["log_dispensary_per_1000"]).yq.nunique()
    print(f"CO quarterly panel: {len(df)} rows, {df.fips5.nunique()} counties, "
          f"periods {df.period.min()}-{df.period.max()}")
    print(f"CO quarters with a genuine (non-snapshot) dispensary count: {n_disp_quarters} "
          f"-- too few for revenue-per-store/density event studies, skipped")

    d = df.dropna(subset=["log_revenue_per_capita", "border_adjacent", "log_population"]).copy()
    formula = ("log_revenue_per_capita ~ border_adjacent + border_adjacent:C(period) + log_population "
               "+ C(period) + C(state_abbr)")
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\nn={len(d)}, {d.fips5.nunique()} counties")
    print(f"Border main effect: b={m.params.get('border_adjacent', float('nan')):+.3f} "
          f"p={m.pvalues.get('border_adjacent', float('nan')):.3f}")

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
    dd.to_csv(os.path.join(OUT, "eventstudy_quarterly_CO_Revenuepercapita.csv"), index=False)
    print(f"{len(dd)} quarterly coefficients estimated")

    apply_healy_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    xt = np.arange(len(dd))
    ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                linewidth=1.2, color=PALETTE["blue"])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    step = max(1, len(dd) // 20)
    ax.set_xticks(xt[::step])
    ax.set_xticklabels(dd["period"].iloc[::step], rotation=90, fontsize=8)
    ax.set_ylabel("Border coefficient (log points)")
    ax.set_title("Border x quarter event study -- Colorado only: Revenue per capita\n"
                  "county-clustered 95% CI", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_CO.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
