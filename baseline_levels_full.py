"""Full baseline comparison: ratio vs levels, log vs asinh, 116 vs 368 ZIPs.
Produces the exact numbers for a rebuilt Table 2."""
import os, numpy as np, pandas as pd, statsmodels.formula.api as smf

BASE = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project"
TBL = os.path.join(BASE, "tables")

d116 = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"), dtype={"zip5": str, "county_fips": str})
d368 = pd.read_csv(os.path.join(TBL, "nm_zip_panel_zeroinclusive.csv"), dtype={"zip5": str, "county_fips": str})

for df in (d116, d368):
    df["log_total_sales"] = np.log(df["total_sales"].where(df["total_sales"] > 0))
    df["log_n_disp"] = np.log(df["n_dispensaries"].where(df["n_dispensaries"] > 0))
    df["asinh_total_sales"] = np.arcsinh(df["total_sales"])
    df["asinh_n_disp"] = np.arcsinh(df["n_dispensaries"])


def fit(df, yv):
    s = df.dropna(subset=[yv, "log_population"]).copy()
    m = smf.ols(f"{yv} ~ border_zip + log_population + C(yq)", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s["county_fips"]})
    star = "".join("*" for t in (0.10, 0.05, 0.01) if m.pvalues["border_zip"] < t)
    gstar = "".join("*" for t in (0.10, 0.05, 0.01) if m.pvalues["log_population"] < t)
    return dict(b=m.params["border_zip"], se=m.bse["border_zip"], p=m.pvalues["border_zip"], star=star,
                g=m.params["log_population"], gse=m.bse["log_population"], gstar=gstar,
                n=int(m.nobs), zips=s["zip5"].nunique(), r2=m.rsquared)


def row(tag, df, yv):
    r = fit(df, yv)
    print(f"{tag:<46} Border {r['b']:+.4f}{r['star']:<4} ({r['se']:.3f}) p={r['p']:.3f}   "
          f"logPop {r['g']:+.3f}{r['gstar']:<4} ({r['gse']:.3f})   n={r['n']}  zips={r['zips']}  R2={r['r2']:.3f}")


print("=== 116-ZIP conditional panel ===")
row("log(revenue/pop)  ~ B + logP   [current prose]", d116, "log_revenue_per_capita")
row("log(revenue)      ~ B + logP   [LEVELS]",        d116, "log_total_sales")
row("log(#disp/pop)    ~ B + logP   [current prose]", d116, "log_dispensary_per_capita")
row("log(#disp)        ~ B + logP   [LEVELS]",        d116, "log_n_disp")
row("log(revenue/disp) ~ B + logP",                   d116, "log_revenue_per_dispensary")
print()
print("=== 368-ZIP zero-inclusive panel ===")
row("asinh(revenue/pop) ~ B + logP  [current TABLE]", d368, "asinh_revenue_per_capita")
row("asinh(revenue)     ~ B + logP  [LEVELS asinh]",  d368, "asinh_total_sales")
row("asinh(#disp/1000)  ~ B + logP  [current TABLE]", d368, "asinh_dispensary_per_capita")
row("asinh(#disp)       ~ B + logP  [LEVELS asinh]",  d368, "asinh_n_disp")
