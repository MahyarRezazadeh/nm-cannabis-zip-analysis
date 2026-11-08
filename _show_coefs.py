import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('tables/master_quarterly_panel_no_wa.csv', dtype={'fips5': str})
border = pd.read_csv('tables/county_border_stateline.csv', dtype={'fips5': str})
df = df.merge(border[['fips5', 'border_county']], on='fips5', how='left')
df['is_capped'] = df['entry_capped']
df['period'] = df['yq']
df['state_period'] = df['state_abbr'] + '_' + df['period'].astype(str)

specs = [('Revenue per capita', 'log_revenue_per_capita'),
         ('Revenue per store', 'log_revenue_per_store'),
         ('Dispensary density', 'log_dispensary_per_1000')]

for label, yvar in specs:
    d = df.dropna(subset=[yvar, 'border_county', 'is_capped']).copy()
    formula = f'{yvar} ~ border_county + border_county:is_capped + C(state_period)'
    m = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['fips5']})
    alpha = m.params['Intercept']
    b = m.params['border_county']
    inter = m.params['border_county:is_capped']
    se_a, se_b, se_i = m.bse['Intercept'], m.bse['border_county'], m.bse['border_county:is_capped']
    p_a, p_b, p_i = m.pvalues['Intercept'], m.pvalues['border_county'], m.pvalues['border_county:is_capped']
    print(f'=== {label} (n={int(m.nobs)}) ===')
    print(f'  alpha (Intercept):        {alpha:+.4f}  (se={se_a:.4f}, p={p_a:.4f})')
    print(f'  beta  (Border):           {b:+.4f}  (se={se_b:.4f}, p={p_b:.4f})  [{(np.exp(b)-1)*100:+.1f}%]')
    print(f'  Border x Capped:          {inter:+.4f}  (se={se_i:.4f}, p={p_i:.4f})  [{(np.exp(inter)-1)*100:+.1f}%]')
    print(f'  R-squared: {m.rsquared:.4f}')
    print()
