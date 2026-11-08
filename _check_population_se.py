import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('tables/nm_zip_quarterly_panel.csv', dtype={'zip5': str, 'county_fips': str})
df['period'] = df['yq']
df['county_fips_clean'] = df['county_fips'].str.replace(r'\.0$', '', regex=True).str.zfill(5)
opt34 = pd.read_csv('tables/nm_zip_border_option34.csv', dtype={'zip5': str})
df = df.drop(columns=['border_zip']).merge(opt34[['zip5', 'border_zip']], on='zip5', how='left')
df['border_zip'] = df['border_zip'].fillna(0).astype(int)

countypop = pd.read_csv('../entry game-marijuana/tables/master_percapita_panel.csv', dtype={'fips5': str})
countypop = countypop[countypop.state_abbr == 'NM'][['fips5', 'population']].drop_duplicates()
countypop = countypop.rename(columns={'fips5': 'county_fips_clean', 'population': 'county_population'})
df = df.merge(countypop, on='county_fips_clean', how='left')
df['log_county_population'] = np.log(df['county_population'])

specs = [('Revenue per capita', 'log_revenue_per_capita'),
         ('Revenue per store', 'log_revenue_per_store'),
         ('Dispensary density', 'log_dispensary_per_1000')]

for label, yvar in specs:
    d = df.dropna(subset=[yvar, 'border_zip', 'log_county_population']).copy()
    m_no = smf.ols(f'{yvar} ~ border_zip + C(period)', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['county_fips_clean']})
    m_yes = smf.ols(f'{yvar} ~ border_zip + log_county_population + C(period)', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['county_fips_clean']})
    b0, se0, p0 = m_no.params['border_zip'], m_no.bse['border_zip'], m_no.pvalues['border_zip']
    b1, se1, p1 = m_yes.params['border_zip'], m_yes.bse['border_zip'], m_yes.pvalues['border_zip']
    print(f'{label}:')
    print(f'  WITHOUT population: b={b0:+.4f} se={se0:.4f} p={p0:.4f}')
    print(f'  WITH population:    b={b1:+.4f} se={se1:.4f} p={p1:.4f}')
    print(f'  SE change: {100*(se1-se0)/se0:+.1f}%   Coef change: {100*(b1-b0)/abs(b0):+.1f}%')
    print()
