"""
Continuous dose-response panels for New Mexico: how do dispensary-level
revenue outcomes and ZIP-level dispensary density respond to distance
from the nonlegal-state line, rather than the binary border/interior
split used elsewhere in this paper?

Distance is binned into 25-mile bands rather than entered as a linear or
polynomial term, to let the data reveal whatever shape the relationship
actually has instead of imposing one. The farthest bin (175-200 miles)
is EXCLUDED entirely, not just used as an omitted reference: 75 percent
of its observations are a single county, Bernalillo County (Albuquerque),
whose ZIP codes average 2.7 times the population of the nearest-border
ZIPs -- an urban-agglomeration confound unrelated to border distance.
The nearest-border bin (0-25 miles) is the reference category instead,
so every reported coefficient is a level relative to dispensaries/ZIPs
immediately adjacent to the nonlegal-state line.

Dispensary-level panel: reuses nm_dispensary_quarterly_panel_v3.csv
(log_revenue_percapita, log_revenue, log_population, period, and each
dispensary's own dist_stateline_mi, already present in that file).

ZIP-level panel: aggregates the same dispensary-quarter data to
(zip5_nearest, quarter), taking log_density_zip (already computed at
this level) and the mean dist_stateline_mi of ALL dispensaries assigned
to that ZIP (computed before any distance-based filtering, so a ZIP's
distance isn't biased by which of its own dispensaries happen to survive
the drop) as the ZIP's own distance.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")

BIN_EDGES = [0, 25, 50, 75, 100, 125, 150, 175, 200]
BIN_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-125", "125-150", "150-175", "175-200"]
DROP_BIN = "175-200"
KEPT_LABELS = [l for l in BIN_LABELS if l != DROP_BIN]  # 0-25 is first -> reference category


def bin_and_drop(df, dist_col):
    df = df.copy()
    df["dist_bin"] = pd.cut(df[dist_col], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True)
    df = df[df["dist_bin"] != DROP_BIN].copy()
    df["dist_bin"] = pd.Categorical(df["dist_bin"], categories=KEPT_LABELS, ordered=False)
    return df


def main():
    raw = pd.read_csv(os.path.join(TBL, "nm_dispensary_quarterly_panel_v3.csv"),
                       dtype={"fips5": str, "zip5_nearest": str})
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw["log_population"] = np.log(raw["population"])

    # ---------- ZIP-level distance: computed from ALL dispensaries (pre-drop) ----------
    zip_dist_mean = raw.groupby("zip5_nearest")["dist_stateline_mi"].mean().reset_index()

    # ---------- dispensary-level panel ----------
    panel = bin_and_drop(raw, "dist_stateline_mi")
    print("Dispensary-level panel: distance bin counts (dispensary-quarters)")
    print(panel["dist_bin"].value_counts().reindex(KEPT_LABELS))

    out_disp = os.path.join(TBL, "nm_dispensary_doseresponse_panel.csv")
    panel.to_csv(out_disp, index=False)
    print(f"Wrote {out_disp}\n")

    # ---------- ZIP-level panel ----------
    zip_panel = raw.drop_duplicates(subset=["zip5_nearest", "period"])[
        ["zip5_nearest", "period", "year", "quarter", "yq", "log_density_zip", "zip_population",
         "n_dispensaries_zip", "fips5"]
    ].copy()
    zip_panel = zip_panel.merge(zip_dist_mean, on="zip5_nearest", how="left")
    zip_panel["log_zip_population"] = np.log(zip_panel["zip_population"])
    zip_panel = zip_panel.replace([np.inf, -np.inf], np.nan)
    zip_panel = bin_and_drop(zip_panel, "dist_stateline_mi")

    print("ZIP-level panel: distance bin counts (ZIP-quarters)")
    print(zip_panel["dist_bin"].value_counts().reindex(KEPT_LABELS))

    out_zip = os.path.join(TBL, "nm_zip_doseresponse_panel.csv")
    zip_panel.to_csv(out_zip, index=False)
    print(f"Wrote {out_zip}")


if __name__ == "__main__":
    main()
