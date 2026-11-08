import pandas as pd
import numpy as np
import patsy

df = pd.read_csv('tables/master_quarterly_panel_no_wa.csv', dtype={'fips5': str})
border = pd.read_csv('tables/county_border_stateline.csv', dtype={'fips5': str})
df = df.merge(border[['fips5', 'border_county']], on='fips5', how='left')
df['is_capped'] = df['entry_capped']
df['period'] = df['yq']

d = df.dropna(subset=['log_revenue_per_capita', 'border_county', 'is_capped', 'log_population'])

formula_full = 'log_revenue_per_capita ~ border_county*is_capped + C(state_abbr) + C(period) + C(state_abbr):C(period)'
y, X = patsy.dmatrices(formula_full, data=d)
rank = np.linalg.matrix_rank(X)
print(f'WITH standalone is_capped: {X.shape[1]} columns, rank {rank}',
      '-- COLLINEAR' if rank < X.shape[1] else '-- full rank')

formula_fixed = 'log_revenue_per_capita ~ border_county + border_county:is_capped + C(state_abbr) + C(period) + C(state_abbr):C(period)'
y2, X2 = patsy.dmatrices(formula_fixed, data=d)
rank2 = np.linalg.matrix_rank(X2)
print(f'WITHOUT standalone is_capped: {X2.shape[1]} columns, rank {rank2}',
      '-- COLLINEAR' if rank2 < X2.shape[1] else '-- full rank')

formula_interaction_only = 'log_revenue_per_capita ~ border_county + border_county:is_capped + C(state_abbr):C(period)'
y3, X3 = patsy.dmatrices(formula_interaction_only, data=d)
rank3 = np.linalg.matrix_rank(X3)
print(f'Interaction-only state x quarter FE: {X3.shape[1]} columns, rank {rank3}',
      '-- COLLINEAR' if rank3 < X3.shape[1] else '-- full rank')

# concatenated state_period key -- only observed combinations get a dummy, avoiding
# phantom empty-cell columns from an unbalanced panel (CO starts 2015, OR 2017, MT/NM 2022)
d = d.copy()
d['state_period'] = d['state_abbr'] + '_' + d['period'].astype(str)
formula_statekey = 'log_revenue_per_capita ~ border_county + border_county:is_capped + C(state_period)'
y4, X4 = patsy.dmatrices(formula_statekey, data=d)
rank4 = np.linalg.matrix_rank(X4)
print(f'state_period concatenated key: {X4.shape[1]} columns, rank {rank4}',
      '-- COLLINEAR' if rank4 < X4.shape[1] else '-- full rank')
print('n unique state_period combos:', d.state_period.nunique())
print('n unique state x n unique period (naive cross product):', d.state_abbr.nunique() * d.period.nunique())
