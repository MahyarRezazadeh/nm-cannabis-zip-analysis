"""
County-level event study, per explicit user request: event time is
QUARTERS SINCE EACH COUNTY'S OWN STATE legalized retail (not calendar
quarter), letting CO (legal since 2014), OR (2015), MT and NM (2022) be
compared on a common clock despite legalizing in different calendar years
-- the same "relative time" alignment already used for the NM+OR
free-period comparison earlier in this project, generalized to all 4
states and split by regime.

Two separate samples/figures: CAPPED regime (CO always, MT always, OR from
its own 2022 cap onward) and FREE regime (NM always, OR before its own
2022 cap) -- an observation is assigned to a regime based on its own
county-quarter's is_capped value, so Oregon contributes to both samples
using the same 2015-10-01 legalization date in each.

Specification: log(y) ~ border_county + border_county:C(rel_q) +
C(rel_q) + C(state_abbr), county-clustered SEs. Each quarter's own level is
border_county + border_county:C(rel_q)[T.k] via t_test (not the raw
interaction coefficient alone -- the same fix applied throughout this
project).
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
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")

RETAIL_START = {"CO": (2014, 1), "OR": (2015, 4), "MT": (2022, 1), "NM": (2022, 2)}  # (year, quarter)


def load():
    df = pd.read_csv(os.path.join(TBL, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
    df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))].copy()
    border = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    df = df.merge(border[["fips5", "border_county"]], on="fips5", how="left")
    df["is_capped"] = df["entry_capped"]

    def rel_q(row):
        y0, q0 = RETAIL_START[row["state_abbr"]]
        return (row["year"] * 4 + row["quarter"]) - (y0 * 4 + q0)

    df["rel_q"] = df.apply(rel_q, axis=1)
    return df


def event_study(df, regime_label, regime_filter, tag):
    d0 = df[regime_filter(df)]
    specs = [("Revenue per capita", "log_revenue_per_capita"),
             ("Revenue per store", "log_revenue_per_store"),
             ("Dispensary density", "log_dispensary_per_1000")]
    es_results = {}
    for label, yvar in specs:
        d = d0.dropna(subset=[yvar, "border_county"]).copy()
        n_states = d.state_abbr.nunique()
        if d["border_county"].nunique() < 2 or len(d) < 30:
            print(f"[{tag}] {label}: skipped (n={len(d)}, insufficient variation)")
            continue
        formula = f"{yvar} ~ border_county + border_county:C(rel_q) + C(rel_q)" + \
                  (" + C(state_abbr)" if n_states > 1 else "")
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

        all_rel_q = sorted(d["rel_q"].unique())
        interacted = {int(t.split("[T.")[1].rstrip("]")) for t in m.params.index
                      if t.startswith("border_county:C(rel_q)")}
        rows = []
        for rq in all_rel_q:
            test_str = (f"border_county + border_county:C(rel_q)[T.{rq}] = 0" if rq in interacted
                        else "border_county = 0")
            tt = m.t_test(test_str)
            rows.append((rq, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
        dd = pd.DataFrame(rows, columns=["rel_q", "coef", "se"])
        es_results[label] = dd
        dd.to_csv(os.path.join(OUT, f"eventstudy_county_{tag}_{label.replace(' ', '')}.csv"), index=False)
        print(f"[{tag}] {label}: n={len(d)}, states={sorted(d.state_abbr.unique())}, "
              f"{len(all_rel_q)} relative quarters")

    if not es_results:
        print(f"[{tag}] no outcomes available -- skipping figure")
        return

    apply_healy_style()
    fig, axes = plt.subplots(1, len(es_results), figsize=(5.5 * len(es_results), 4.5), squeeze=False)
    axes = axes[0]
    for ax, (label, dd) in zip(axes, es_results.items()):
        step = max(1, len(dd) // 15)
        xt = np.arange(len(dd))
        ax.errorbar(xt, dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=2, markersize=3,
                    linewidth=1, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt[::step])
        ax.set_xticklabels(dd["rel_q"].iloc[::step], rotation=90, fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Quarters since own-state legalization")
        ax.set_ylabel("Border coefficient (log points)")
    fig.suptitle(f"County border x event-time study -- {regime_label} regime, county-clustered 95% CI",
                 fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT, f"fig_eventstudy_county_{tag}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    print(f"{len(df)} county-quarter rows. RETAIL_START used: {RETAIL_START}\n")
    event_study(df, "Capped", lambda d: d.is_capped == 1, "capped")
    event_study(df, "Free", lambda d: d.is_capped == 0, "free")
