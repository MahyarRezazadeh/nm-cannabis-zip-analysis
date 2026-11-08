"""
Rebuild New Mexico's dispensary-level panel keeping ZIP code (not just
county), from the same 51 monthly CROP licensee files already used for
nm_dispensary_panel_monthly.csv. That earlier file only kept county
assignment; this one keeps zip5 directly (no ZCTA crosswalk needed for
NM, since Zip5 is reported directly on each licensee record), for a true
ZIP-code-level quarterly panel per explicit user request.

Dispensary identity = normalized street address, same convention as
nm_dispensary_panel_monthly.csv.
"""
import glob
import os
import re
import pandas as pd

SOURCE_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "entry game-marijuana")
RAW_DIR = os.path.join(SOURCE_BASE, "data", "raw", "newmexico_revenue")
BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")

MONTHS = {m: i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"])}


def parse_year_month(filename):
    base = os.path.basename(filename)
    year = int(re.match(r"(\d{4})_", base).group(1))
    month = None
    for m in MONTHS:
        if m in base:
            month = MONTHS[m]
            break
    if month is None:
        raise ValueError(f"Could not parse month from {base}")
    return year, month


def normalize_address(addr):
    if pd.isna(addr):
        return None
    a = str(addr).strip().upper()
    a = re.sub(r"\s+", " ", a)
    return a.rstrip(".")


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*Licensee*")))
    print(f"{len(files)} licensee files found")

    frames = []
    for f in files:
        year, month = parse_year_month(f)
        df = pd.read_excel(f)
        keep = ["Licensee", "Address", "City", "State", "Zip", "Medical Sales", "Adult-Use Sales", "Total Sales"]
        df = df[[c for c in keep if c in df.columns]].dropna(subset=["Licensee"])
        df = df[~df["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().eq("totals")]
        df["Total Sales"] = pd.to_numeric(df["Total Sales"], errors="coerce").fillna(0)
        df["year"], df["month"] = year, month
        df["zip5"] = df["Zip"].astype(str).str.extract(r"(\d{5})")[0]
        df["addr_norm"] = df["Address"].apply(normalize_address)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["addr_norm", "zip5"])

    disp_id = full[["addr_norm"]].drop_duplicates().reset_index(drop=True)
    disp_id["dispensary_id"] = disp_id.index
    full = full.merge(disp_id, on="addr_norm", how="left")

    # most frequently observed zip5 per dispensary_id (a handful of stores report a typo'd zip
    # in one month and the correct one elsewhere; take the mode, not the first-seen value)
    zip_mode = (full.groupby(["dispensary_id", "zip5"]).size().reset_index(name="n")
                .sort_values("n", ascending=False).drop_duplicates("dispensary_id")[["dispensary_id", "zip5"]])
    full = full.drop(columns=["zip5"]).merge(zip_mode, on="dispensary_id", how="left")

    monthly = (full.groupby(["dispensary_id", "zip5", "year", "month"])
               .agg(total_sales=("Total Sales", "sum")).reset_index())

    os.makedirs(TBL, exist_ok=True)
    out = os.path.join(TBL, "nm_dispensary_panel_monthly_zip.csv")
    monthly.to_csv(out, index=False)
    print(f"Wrote {out}: {monthly.shape}, {monthly.dispensary_id.nunique()} unique dispensaries, "
          f"{monthly.zip5.nunique()} unique ZIP codes")


if __name__ == "__main__":
    main()
