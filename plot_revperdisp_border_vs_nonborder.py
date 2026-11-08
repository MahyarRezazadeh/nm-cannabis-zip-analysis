"""
Descriptive graph: revenue per dispensary, border vs non-border New Mexico
dispensaries, quarterly 2022Q2-2026Q2 (2026Q3 dropped as a partial quarter).

Dispensary-level data: tables/nm_dispensary_quarterly_panel_v3.csv
  - total_sales    : the dispensary's own quarterly revenue (= revenue per dispensary)
  - border_dispensary : 1 if the dispensary's own geocoded location is within 75 mi
                        of the TX/OK/UT state line with >=1,000 nonlegal residents
                        within 75 mi (predetermined, geography-only; same rule as the
                        ZIP-level design, applied without the ZIP-aggregation step)

Two panels:
  (a) arithmetic mean revenue per dispensary (= total group revenue / # dispensaries),
      with 95% CI from the group SE each quarter
  (b) geometric mean revenue per dispensary (exp of the mean of log revenue) -- the
      quantity the paper's log-revenue regressions actually compare -- with 95% CI

Not a regression. The fitted border coefficient on log revenue per dispensary in the
noise-minimizing dispensary-level spec is +16.2% (p=0.144, county-clustered): a
visible but not statistically robust gap, which is what this figure shows.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")
sys.path.insert(0, os.path.join(BASE, "..", "Latest try", "code"))
try:
    from healy_style import apply_healy_style, PALETTE
    apply_healy_style()
    C_BORDER, C_INT = PALETTE["red"], PALETTE["blue"]
except Exception:
    C_BORDER, C_INT = "#b2182b", "#2166ac"

DROP_Q = {"2026Q3"}  # partial quarter


def main():
    d = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"), dtype={"fips5": str})
    d = d[~d["yq"].isin(DROP_Q)].copy()
    d = d[d["total_sales"] > 0].copy()
    d["logrev"] = np.log(d["total_sales"])

    def summarise(g):
        n = g["total_sales"].notna().sum()
        m = g["total_sales"].mean()
        se = g["total_sales"].std(ddof=1) / np.sqrt(n)
        lm = g["logrev"].mean()
        lse = g["logrev"].std(ddof=1) / np.sqrt(n)
        return pd.Series({"n": n, "mean": m, "mean_lo": m - 1.96 * se, "mean_hi": m + 1.96 * se,
                          "gmean": np.exp(lm), "gmean_lo": np.exp(lm - 1.96 * lse),
                          "gmean_hi": np.exp(lm + 1.96 * lse)})

    s = d.groupby(["yq", "border_dispensary"]).apply(summarise).reset_index()
    s.to_csv(os.path.join(OUT, "revperdisp_border_vs_nonborder_quarterly.csv"), index=False)
    print(s.to_string())

    quarters = sorted(s["yq"].unique())
    x = np.arange(len(quarters))
    xi = {q: i for i, q in enumerate(quarters)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
    for ax, (col, lo, hi, ttl) in zip(
        axes,
        [("mean", "mean_lo", "mean_hi", "Mean revenue per dispensary"),
         ("gmean", "gmean_lo", "gmean_hi", "Geometric mean revenue per dispensary\n(the quantity the log-revenue regressions compare)")],
    ):
        for bd, colr, lab in [(1, C_BORDER, "Border (<=75 mi from nonlegal-state line)"),
                              (0, C_INT, "Interior")]:
            sub = s[s["border_dispensary"] == bd].sort_values("yq")
            xs = [xi[q] for q in sub["yq"]]
            ax.plot(xs, sub[col] / 1000, "o-", color=colr, markersize=4, linewidth=1.5, label=lab)
            ax.fill_between(xs, sub[lo] / 1000, sub[hi] / 1000, color=colr, alpha=0.15)
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel("Revenue per dispensary ($000, nominal)")
        ax.set_xticks(x)
        ax.set_xticklabels(quarters, rotation=90, fontsize=8)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=9, loc="upper right", frameon=False)
    fig.suptitle("Revenue per dispensary: border vs. interior New Mexico dispensaries, 2022Q2-2026Q2\n"
                 "dispensary-level panel (n=1,192 dispensaries); shaded = 95% CI; not a regression",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_revperdisp_border_vs_nonborder.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("\nWrote", out)


if __name__ == "__main__":
    main()
