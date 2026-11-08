"""
Event study of the CATCHMENT-POPULATION control itself, per explicit user
request: extends the "+ catchment population" specification
(analyze_eventstudy_quarterly_NM_zip_v2.py) by also interacting
log_population_within_70mi with C(period), so its own coefficient -- not
just border_zip's -- is allowed to vary by quarter. Both border_zip's and
log_population_within_70mi's period-specific levels are extracted the same
way (main effect + interaction via t_test, covariance-adjusted SE).

Specification: log(y) ~ border_zip + border_zip:C(period) + log_population
+ log_population_within_70mi + log_population_within_70mi:C(period) +
C(period) + C(state_abbr) + C(state_abbr):C(period), clustered by county.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "figures and tables")


def period_levels(m, d, var, prefix):
    all_periods = sorted(d["period"].unique())
    interacted = {t.split("[T.")[1].rstrip("]") for t in m.params.index if t.startswith(prefix)}
    rows = []
    for p in all_periods:
        test_str = f"{var} + {prefix}[T.{p}] = 0" if p in interacted else f"{var} = 0"
        tt = m.t_test(test_str)
        rows.append((p, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])))
    return pd.DataFrame(rows, columns=["period", "coef", "se"])


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(TBL, "nm_zip_quarterly_panel.csv"), dtype={"zip5": str, "county_fips": str})
    df["period"] = df["yq"]
    catch = pd.read_csv(os.path.join(TBL, "nm_zip_catchment_population.csv"), dtype={"zip5": str})
    df = df.merge(catch[["zip5", "population_within_70mi"]], on="zip5", how="left")
    df["log_population_within_70mi"] = np.log(df["population_within_70mi"])

    specs = [
        ("Revenue per capita", "log_revenue_per_capita"),
        ("Revenue per store", "log_revenue_per_store"),
        ("Dispensary density", "log_dispensary_per_1000"),
    ]

    for label, yvar in specs:
        d = df.dropna(subset=[yvar, "border_zip", "log_population", "log_population_within_70mi"]).copy()
        formula = (f"{yvar} ~ border_zip + border_zip:C(period) + log_population "
                   f"+ log_population_within_70mi + log_population_within_70mi:C(period) "
                   f"+ C(period) + C(state_abbr) + C(state_abbr):C(period)")
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["county_fips"]})

        border_dd = period_levels(m, d, "border_zip", "border_zip:C(period)")
        catch_dd = period_levels(m, d, "log_population_within_70mi", "log_population_within_70mi:C(period)")

        border_dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_zip_v2b_border_{label.replace(' ', '')}.csv"),
                          index=False)
        catch_dd.to_csv(os.path.join(OUT, f"eventstudy_quarterly_NM_zip_v2b_catchment_{label.replace(' ', '')}.csv"),
                         index=False)
        print(f"{label}: n={len(d)}")
        print(f"  border_zip reference-period effect: b={m.params.get('border_zip', float('nan')):+.3f} "
              f"p={m.pvalues.get('border_zip', float('nan')):.3f}")
        print(f"  catchment-pop reference-period effect: "
              f"b={m.params.get('log_population_within_70mi', float('nan')):+.3f} "
              f"p={m.pvalues.get('log_population_within_70mi', float('nan')):.3f}")


if __name__ == "__main__":
    main()
