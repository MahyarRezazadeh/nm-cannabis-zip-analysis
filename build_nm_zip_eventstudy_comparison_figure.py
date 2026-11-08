"""
Combined event-study comparison figure for the two DISTINCT specifications
in table_nm_zip_maintable_comparison (Original and State/qtr/state x qtr FE
are numerically identical for this NM-only panel, so shown as one row):
  Row 1: Original / State-quarter-FE spec (border_zip + log_population +
         C(period) + C(state_abbr) [+ C(state_abbr):C(period), a no-op])
  Row 2: + population-within-70mi catchment control
Reuses the already-fitted per-period coefficients saved by
analyze_eventstudy_quarterly_NM_zip.py and _v2.py -- no re-estimation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "figures and tables")

OUTCOMES = ["Revenuepercapita", "Revenueperstore", "Dispensarydensity"]
TITLES = {"Revenuepercapita": "Revenue per capita", "Revenueperstore": "Revenue per store",
          "Dispensarydensity": "Dispensary density"}
ROWS = [("Original (= state/qtr/state x qtr FE)\nborder coefficient",
         "eventstudy_quarterly_NM_zip_{}.csv", PALETTE["blue"]),
        ("+ catchment population (within 70mi)\nborder coefficient",
         "eventstudy_quarterly_NM_zip_v2_{}.csv", PALETTE["blue"]),
        ("+ catchment population (within 70mi)\ncatchment-population coefficient itself",
         "eventstudy_quarterly_NM_zip_v2b_catchment_{}.csv", PALETTE["red"])]


def main():
    apply_healy_style()
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), sharex="col")
    for row_i, (row_label, fname_tmpl, color) in enumerate(ROWS):
        for col_i, outcome in enumerate(OUTCOMES):
            dd = pd.read_csv(os.path.join(OUT, fname_tmpl.format(outcome)))
            ax = axes[row_i, col_i]
            xt = np.arange(len(dd))
            ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, markersize=4,
                        linewidth=1.2, color=color)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xticks(xt)
            ax.set_xticklabels(dd["period"], rotation=90, fontsize=7)
            if row_i == 0:
                ax.set_title(TITLES[outcome], fontsize=11)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\n\nCoefficient (log points)", fontsize=9)
    fig.suptitle("NM ZIP-code border x quarter event study: specification comparison\n"
                  "county-clustered 95% CI", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(OUT, "fig_eventstudy_quarterly_NM_zip_speccomparison.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
