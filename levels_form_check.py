"""
Re-run the baseline (eq. 1) and dynamic (eq. 2) border-premium specs in LEVELS
form -- outcome = log(revenue), log(dispensary count) -- with log(own ZIP
population) as a free right-hand-side regressor, and compare to the current
per-capita / density (ratio) form.

Algebra: log(R/P) = ... + gamma*log(P)  <=>  log(R) = ... + (1+gamma)*log(P),
so every coefficient except the one on log(P) should be identical.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project"
TBL = os.path.join(BASE, "tables")
SP = r"C:\Users\mahya\AppData\Local\Temp\claude\C--Users-mahya\ff85abd8-d475-4e64-b705-5f9059643e0c\scratchpad"
REF = "2022Q2"

d = pd.read_csv(os.path.join(TBL, "nm_zip_panel_corrected.csv"), dtype={"zip5": str, "county_fips": str})
d["log_total_sales"] = np.log(d["total_sales"].where(d["total_sales"] > 0))
d["log_n_dispensaries"] = np.log(d["n_dispensaries"].where(d["n_dispensaries"] > 0))

# (outcome pairs): ratio form vs levels form
PAIRS = [
    ("Revenue per capita",      "log_revenue_per_capita",     "log_total_sales"),
    ("Dispensary density",      "log_dispensary_per_capita",   "log_n_dispensaries"),
    ("Revenue per dispensary",  "log_revenue_per_dispensary",  "log_revenue_per_dispensary"),  # no P in outcome
]


def baseline(yvar):
    s = d.dropna(subset=[yvar, "log_population"]).copy()
    m = smf.ols(f"{yvar} ~ border_zip + log_population + C(yq)", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s["county_fips"]})
    return (m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"],
            m.params["log_population"], int(m.nobs))


print("=" * 96)
print("BASELINE (eq. 1)   -- Border coefficient, county-clustered SE")
print("=" * 96)
print(f"{'outcome':<24}{'form':<10}{'Border b':>12}{'se':>9}{'p':>8}{'coef on log(pop)':>20}{'n':>7}")
for lab, yr, yl in PAIRS:
    br, ser, pr, gr, nr = baseline(yr)
    print(f"{lab:<24}{'ratio':<10}{br:>+12.4f}{ser:>9.4f}{pr:>8.3f}{gr:>20.4f}{nr:>7}")
    if yl != yr:
        bl, sel, pl, gl, nl = baseline(yl)
        print(f"{'':<24}{'levels':<10}{bl:>+12.4f}{sel:>9.4f}{pl:>8.3f}{gl:>20.4f}{nl:>7}")
        print(f"{'':<24}{'diff':<10}{bl - br:>+12.6f}{'':9}{'':8}{'(gamma vs 1+gamma)':>20}")
    print("-" * 96)


def dynamic(yvar):
    s = d.dropna(subset=[yvar, "log_population"]).copy()
    qs = sorted(s["yq"].unique())
    f = f"{yvar} ~ border_zip*C(yq, Treatment(reference='{REF}')) + log_population"
    m = smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s["county_fips"]})
    out = []
    for q in qs:
        if q == REF:
            b, se = m.params["border_zip"], m.bse["border_zip"]
        else:
            t = f"border_zip:C(yq, Treatment(reference='{REF}'))[T.{q}]"
            if t not in m.params.index:
                continue
            tt = m.t_test(f"border_zip + {t}")
            b, se = float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])
        out.append((q, b, se))
    return pd.DataFrame(out, columns=["yq", "coef", "se"])


print("\n" + "=" * 96)
print("DYNAMIC (eq. 2)  -- border-by-quarter coefficients: ratio form vs levels form")
print("=" * 96)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (lab, yr, yl) in zip(axes, PAIRS):
    er = dynamic(yr)
    if yl != yr:
        el = dynamic(yl)
        mrg = er.merge(el, on="yq", suffixes=("_ratio", "_levels"))
        maxdiff = (mrg["coef_ratio"] - mrg["coef_levels"]).abs().max()
        maxdiff_se = (mrg["se_ratio"] - mrg["se_levels"]).abs().max()
        print(f"{lab:<24} max |coef diff| over {len(mrg)} quarters = {maxdiff:.2e} ;"
              f"  max |se diff| = {maxdiff_se:.2e}")
        x = np.arange(len(er))
        ax.errorbar(x - 0.12, er["coef"], yerr=1.96 * er["se"], fmt="o-", capsize=2, ms=4,
                    lw=1.1, color="#c0392b", label="per-capita / density (current)")
        ax.errorbar(x + 0.12, el["coef"], yerr=1.96 * el["se"], fmt="s--", capsize=2, ms=3.5,
                    lw=1.0, color="#2c6fbb", label="levels + log(population)")
        ax.set_xticks(x); ax.set_xticklabels(er["yq"], rotation=90, fontsize=7)
    else:
        print(f"{lab:<24} outcome has no population term; spec unchanged.")
        x = np.arange(len(er))
        ax.errorbar(x, er["coef"], yerr=1.96 * er["se"], fmt="o-", capsize=2, ms=4,
                    lw=1.1, color="#c0392b", label="unchanged")
        ax.set_xticks(x); ax.set_xticklabels(er["yq"], rotation=90, fontsize=7)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(lab, fontsize=10); ax.set_xlabel("Quarter")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Border coefficient (log points)")
fig.suptitle("Dynamic border premium: current ratio form vs. levels form with log(population)\n"
             "New Mexico, reference quarter 2022Q2, county-clustered 95% CI", fontsize=11)
fig.tight_layout()
p = os.path.join(SP, "levels_vs_ratio_eventstudy.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"\nWrote {p}")

# also: full levels decomposition for revenue per dispensary (free coef on log N)
print("\n" + "=" * 96)
print("Revenue per dispensary -- full levels decomposition (frees the coef on log N)")
print("=" * 96)
s = d.dropna(subset=["log_total_sales", "log_n_dispensaries", "log_population"]).copy()
m = smf.ols("log_total_sales ~ border_zip + log_population + log_n_dispensaries + C(yq)",
           data=s).fit(cov_type="cluster", cov_kwds={"groups": s["county_fips"]})
print(f"  log(revenue) ~ Border + log(pop) + log(N):  Border b = {m.params['border_zip']:+.4f} "
      f"({m.bse['border_zip']:.4f}), p={m.pvalues['border_zip']:.3f};  "
      f"coef on log(N) = {m.params['log_n_dispensaries']:.3f} (vs 1 in the ratio form)")
