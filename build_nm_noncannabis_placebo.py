"""
Non-cannabis-sector placebo check for the border premium: if elevated
revenue near the nonlegal-state line is specific to cannabis demand
crossing the border, sectors unrelated to cannabis policy should show
no comparable border premium. We test this using New Mexico's own
Taxation and Revenue Department quarterly Gross Receipts Tax reports
(RP-80: "Gross Receipts by Geographic Area and NAICS Code"), which
report gross receipts by NAICS industry for ~260 sub-county reporting
locations (incorporated municipalities plus each county's unincorporated
"Remainder"), 2022Q2-2026Q2 -- the same period as the paper's own NM
dispensary panel.

Three outcomes, chosen to trade off specificity against small-cell tax
suppression (very small border towns often have a "*" suppressed cell
for a narrow 6-digit NAICS code):
  - Alcohol retail: NAICS 445310, "Beer, Wine, and Liquor Stores" -- the
    closest non-cannabis analog to a cannabis dispensary (same general
    function: a licensed retailer of a controlled recreational substance).
  - Grocery stores: NAICS 4451 -- a much larger, rarely-suppressed
    category, useful as a check that isn't underpowered by suppression.
  - Total retail trade: NAICS sector 44-45 -- the broadest placebo,
    testing whether border areas simply have more retail activity of
    any kind (tourism, highway commerce) rather than specifically more
    cannabis activity.

Border status is assigned at the COUNTY level (this project's existing
75-mile county_border_stateline.csv classification), since NM's tax
reporting locations don't carry their own lat/lon -- each reporting
location's leading two-digit code identifies its home county (verified
empirically from the ~33 "County, Remainder" rows, one per county, which
are the only rows with an explicit county name in the location string).
"""
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import openpyxl
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_BASE = os.path.join(os.path.dirname(BASE), "entry game-marijuana")
TBL = os.path.join(BASE, "tables")
RAW = os.path.join(BASE, "data", "nm_grt_raw")

TARGET_NAICS = {
    "445310": "alcohol",
    "4451": "grocery",
    "44": "total_retail",
}


def parse_quarter_from_filename(path):
    m = re.search(r"(\d{4}) Q(\d)", os.path.basename(path))
    year, q = int(m.group(1)), int(m.group(2))
    return f"{year}Q{q}"


def parse_money(v):
    if v is None or v == "*":
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).replace("$", "").replace(",", ""))


def load_one_file(path):
    # NM's report generator changed NAICS label text partway through the sample
    # (2024+ files reflect the 2022 NAICS revision: "44 - Sector 44-45 -- Retail
    # Trade (USA/CAN/MEX)" became "44 - Retail Trade", "4451 - Grocery Stores"
    # became "4451 - Grocery and Convenience Retailers", and 445310 lost its
    # description text entirely, appearing as a bare code). Matching on the
    # leading NAICS code (before " - ", or the whole string if there's no
    # separator) instead of the full description string is robust to this.
    quarter = parse_quarter_from_filename(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = []
    for row in ws.iter_rows(values_only=True):
        if not row or not row[2]:
            continue
        code = str(row[2]).split(" - ")[0].strip()
        if code not in TARGET_NAICS:
            continue
        loc = row[0]
        rows.append({
            "quarter": quarter,
            "location": loc,
            "sector": TARGET_NAICS[code],
            "count": row[3],
            "gross_receipts": parse_money(row[4]),
        })
    wb.close()
    return pd.DataFrame(rows)


def build_location_county_map(all_locations):
    remainder_re = re.compile(r"^(\d{2})\d{3} - (.+) County, Remainder$")
    code_to_county = {}
    for loc in all_locations:
        m = remainder_re.match(loc)
        if m:
            code_to_county[m.group(1)] = m.group(2)
    print(f"Found {len(code_to_county)} county codes from 'County, Remainder' rows")

    loc_county = {}
    unmatched = []
    for loc in all_locations:
        code = loc[:2]
        if code in code_to_county:
            loc_county[loc] = code_to_county[code]
        else:
            unmatched.append(loc)
    if unmatched:
        print(f"WARNING: {len(unmatched)} locations had no matching county code, e.g. {unmatched[:5]}")
    return loc_county


def normalize_county(name):
    name = name.replace("Ã±", "n").replace("ñ", "n")
    name = re.sub(r"\s+County$", "", name)
    return name.strip().upper()


def run_calendar_quarter_event_study(county_q, sector, out_dir):
    """
    Border_c interacted with EVERY calendar quarter (not relative-to-event
    time, since alcohol/grocery/retail never experience a policy change in
    this sample -- there is no legalization date to build relative-quarter
    dummies around). This tests whether the pooled null border coefficient
    reported in the main table is stable across all 17 quarters, or whether
    it is hiding a period where a gap opens up.
    """
    d = county_q[county_q.sector == sector].dropna(subset=["gross_receipts", "log_population"]).copy()
    d = d[d.gross_receipts > 0]
    quarters = sorted(d["quarter"].unique())
    ref_q = quarters[0]

    formula = (f"log_gross_receipts ~ border_county*C(quarter, Treatment(reference='{ref_q}')) "
               f"+ log_population")
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

    rows = []
    for q in quarters:
        if q == ref_q:
            b, se, p = m.params["border_county"], m.bse["border_county"], m.pvalues["border_county"]
        else:
            term = f"border_county:C(quarter, Treatment(reference='{ref_q}'))[T.{q}]"
            if term not in m.params.index:
                continue
            tt = m.t_test(f"border_county + {term}")
            b = float(np.ravel(tt.effect)[0])
            se = float(np.ravel(tt.sd)[0])
            p = float(np.ravel(tt.pvalue)[0])
        rows.append({"sector": sector, "quarter": q, "coef": b, "se": se, "p": p})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(out_dir, f"nm_noncannabis_placebo_eventstudy_{sector}.csv"), index=False)
    n_sig = (ev["p"] < 0.05).sum()
    print(f"{sector}: {n_sig}/{len(ev)} quarters significant at p<0.05, "
          f"coef range [{ev['coef'].min():+.2f}, {ev['coef'].max():+.2f}]")
    return ev


def main():
    files = sorted(glob.glob(os.path.join(RAW, "rp80 *.xlsx")))
    print(f"Found {len(files)} RP-80 files")

    frames = []
    for f in files:
        print(f"Reading {os.path.basename(f)}...")
        frames.append(load_one_file(f))
    panel = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(panel)}")

    loc_county = build_location_county_map(panel["location"].unique())
    panel["county_raw"] = panel["location"].map(loc_county)
    panel = panel.dropna(subset=["county_raw"]).copy()
    panel["county_key"] = panel["county_raw"].apply(normalize_county)

    border = pd.read_csv(os.path.join(TBL, "county_border_stateline.csv"), dtype={"fips5": str})
    nm_border = border[border.state_abbr == "NM"].copy()
    nm_border["county_key"] = nm_border["county_name"].apply(normalize_county)

    panel = panel.merge(nm_border[["county_key", "fips5", "border_county"]], on="county_key", how="left")
    missing = panel[panel.fips5.isna()]["county_raw"].unique()
    if len(missing):
        print(f"WARNING: unmatched counties: {missing}")
    panel = panel.dropna(subset=["fips5"]).copy()

    out_path = os.path.join(TBL, "nm_noncannabis_placebo_panel_location.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}, {len(panel)} rows (location-quarter-sector)")
    print(panel.groupby("sector")["gross_receipts"].apply(lambda s: s.notna().mean()))

    # ---------- aggregate location-quarter to county-quarter ----------
    # Sub-county "reporting locations" have no population of their own, so a location-
    # level regression without a size control is confounded: border counties include
    # Dona Ana (Las Cruces) and San Juan (Farmington), NM's 2nd- and among its largest
    # cities, which mechanically report far higher raw receipts in EVERY sector than
    # small interior "Remainder" locations regardless of any border effect. Aggregating
    # to county-quarter and controlling for log(population) -- the same size control
    # used in every other border-premium regression in this paper -- removes that
    # population confound.
    county_q = panel.groupby(["fips5", "quarter", "sector"], as_index=False)["gross_receipts"].sum()

    zips = pd.read_csv(os.path.join(SOURCE_BASE, "data", "raw", "uszips_simplemaps.csv"),
                        dtype={"zip": str, "county_fips": str})
    nm_pop = zips[zips.state_id == "NM"].groupby("county_fips")["population"].sum().reset_index()
    nm_pop = nm_pop.rename(columns={"county_fips": "fips5"})

    county_q = county_q.merge(nm_pop, on="fips5", how="left")
    county_q = county_q.merge(nm_border[["fips5", "border_county"]].drop_duplicates(), on="fips5", how="left")
    county_q["log_gross_receipts"] = np.log1p(county_q["gross_receipts"])
    county_q["log_population"] = np.log(county_q["population"])

    out_path2 = os.path.join(TBL, "nm_noncannabis_placebo_panel_county.csv")
    county_q.to_csv(out_path2, index=False)
    print(f"Wrote {out_path2}, {len(county_q)} rows (county-quarter-sector)")

    # ---------- border-premium regression per sector, county-quarter, pop control ----------
    print("\n=== Border premium by sector (county-quarter panel, log-population control) ===")
    results = []
    for sector in TARGET_NAICS.values():
        d = county_q[county_q.sector == sector].dropna(subset=["gross_receipts", "log_population"]).copy()
        d = d[d.gross_receipts > 0]
        formula = "log_gross_receipts ~ border_county + log_population + C(quarter)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        b, se, p = m.params["border_county"], m.bse["border_county"], m.pvalues["border_county"]
        pct = (np.exp(b) - 1) * 100
        results.append({"sector": sector, "coef": b, "se": se, "p": p, "pct": pct,
                         "n": int(m.nobs), "n_counties": d["fips5"].nunique()})
        print(f"{sector}: b={b:+.3f} ({se:.3f}), {pct:+.1f}%, p={p:.3f}, n={int(m.nobs)}, "
              f"counties={d['fips5'].nunique()}")

    pd.DataFrame(results).to_csv(os.path.join(TBL, "nm_noncannabis_placebo_results.csv"), index=False)

    # ---------- calendar-quarter event study per sector ----------
    OUT = os.path.join(BASE, "figures and tables")
    print("\n=== Calendar-quarter event study: is the null stable across all 17 quarters? ===")
    ev_all = {}
    for sector in TARGET_NAICS.values():
        ev_all[sector] = run_calendar_quarter_event_study(county_q, sector, TBL)

    apply_healy_style()
    labels = {"alcohol": "Alcohol retail (445310)", "grocery": "Grocery stores (4451)",
              "total_retail": "Total retail trade (44-45)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    for ax, sector in zip(axes, TARGET_NAICS.values()):
        ev = ev_all[sector]
        xt = np.arange(len(ev))
        ax.errorbar(xt, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", capsize=3,
                    markersize=4, linewidth=1.2, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(xt)
        ax.set_xticklabels(ev["quarter"], rotation=90, fontsize=7)
        ax.set_title(labels[sector], fontsize=10)
        ax.set_xlabel("Quarter")
    axes[0].set_ylabel("Border coefficient (log points)")
    fig.suptitle("Border coefficient by calendar quarter, non-cannabis sectors, New Mexico\n"
                 "County-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(OUT, "fig_nm_noncannabis_placebo_eventstudy.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {fig_path}")


if __name__ == "__main__":
    main()
