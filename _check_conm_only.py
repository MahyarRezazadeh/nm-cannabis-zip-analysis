import pandas as pd
import numpy as np
import patsy
import statsmodels.formula.api as smf

df = pd.read_csv('tables/master_quarterly_panel_no_wa.csv', dtype={'fips5': str})
border = pd.read_csv('tables/county_border_stateline.csv', dtype={'fips5': str})
df = df.merge(border[['fips5', 'border_county']], on='fips5', how='left')
df['is_capped'] = df['entry_capped']
df['period'] = df['yq']
df['state_period'] = df['state_abbr'] + '_' + df['period'].astype(str)

conm = df[df.state_abbr.isin(['CO', 'NM'])].copy()

specs = [('Revenue per store', 'log_revenue_per_store'),
         ('Dispensary density', 'log_dispensary_per_1000')]

for label, yvar in specs:
    d = conm.dropna(subset=[yvar, 'border_county', 'is_capped']).copy()
    formula = f'{yvar} ~ border_county + border_county:is_capped + C(state_period)'
    y, X = patsy.dmatrices(formula, data=d)
    rank = np.linalg.matrix_rank(X)
    print(f'{label}: n={len(d)}, states={sorted(d.state_abbr.unique())}, '
          f'design matrix {X.shape[1]} cols, rank {rank}', '-- COLLINEAR' if rank < X.shape[1] else '-- full rank')

    m = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['fips5']})
    b = m.params.get('border_county', np.nan)
    p = m.pvalues.get('border_county', np.nan)
    inter = m.params.get('border_county:is_capped', np.nan)
    p_i = m.pvalues.get('border_county:is_capped', np.nan)
    print(f'  border: {b:+.4f} (p={p:.4f}) [{(np.exp(b)-1)*100:+.1f}%]')
    print(f'  border x capped: {inter:+.4f} (p={p_i:.4f}) [{(np.exp(inter)-1)*100:+.1f}%]')
    print()
