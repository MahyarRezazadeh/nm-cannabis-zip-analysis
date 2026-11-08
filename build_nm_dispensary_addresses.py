"""
Re-extract full address (street + city + state + zip) per NM dispensary,
for Census geocoding. Recomputes dispensary_id fresh from addr_norm here
(identical normalization to build_nm_zip_dispensary_panel.py) rather than
joining on dispensary_id across files -- that ID is just a row-order index
assigned independently in each build script and is NOT guaranteed to mean
the same dispensary across files.
"""
import glob
import os
import re
import pandas as pd

SOURCE_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "entry game-marijuana")
RAW_DIR = os.path.join(SOURCE_BASE, "data", "raw", "newmexico_revenue")
BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")


def normalize_address(addr):
    if pd.isna(addr):
        return None
    a = str(addr).strip().upper()
    a = re.sub(r"\s+", " ", a)
    return a.rstrip(".")


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*Licensee*")))
    frames = []
    for f in files:
        df = pd.read_excel(f)
        keep = ["Licensee", "Address", "City", "State", "Zip"]
        df = df[[c for c in keep if c in df.columns]].dropna(subset=["Licensee"])
        df = df[~df["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().eq("totals")]
        df["zip5"] = df["Zip"].astype(str).str.extract(r"(\d{5})")[0]
        df["addr_norm"] = df["Address"].apply(normalize_address)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["addr_norm", "zip5"])

    disp_id = full[["addr_norm"]].drop_duplicates().reset_index(drop=True)
    disp_id["dispensary_id"] = disp_id.index
    full = full.merge(disp_id, on="addr_norm", how="left")

    # most frequently observed city/zip per dispensary (a few report a typo'd zip in one month)
    mode_row = (full.groupby(["dispensary_id", "addr_norm", "City", "State", "zip5"])
                .size().reset_index(name="n")
                .sort_values("n", ascending=False).drop_duplicates("dispensary_id"))
    result = mode_row[["dispensary_id", "addr_norm", "City", "State", "zip5"]]

    out_path = os.path.join(TBL, "nm_dispensary_addresses.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(result)} unique dispensary addresses")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
