"""Recompute descriptive coverage and error tables from frozen observations/predictions."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-packaged', action='store_true',
                        help='Explicitly replace results/diagnostics; default writes to recomputed/diagnostics.')
    args = parser.parse_args()
    dest = ROOT / ('results/diagnostics' if args.refresh_packaged else 'recomputed/diagnostics')
    dest.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(ROOT / 'data/annual/company_financials.csv')
    quarter = pd.read_csv(ROOT / 'data/quarterly/company_quarterly.csv')
    coverage = annual.groupby('company', sort=True).agg(rows=('year', 'size'),
                  first_year=('year', 'min'), last_year=('year', 'max'),
                  eligible_rows=('main_eligible', 'sum'), currency=('currency', 'first'),
                  sector=('sector', 'first')).reset_index()
    coverage.to_csv(dest / 'annual_coverage.csv', index=False)
    fields = ['sales_revenue', 'reported_revenue', 'net_profit', 'assets', 'equity']
    missing = pd.DataFrame({'field': fields, 'missing_count': annual[fields].isna().sum().to_numpy(),
                           'rows': len(annual)})
    missing['missing_share'] = missing.missing_count / missing.rows
    missing.to_csv(dest / 'annual_missingness.csv', index=False)
    eligible = annual[annual.main_eligible].copy()
    eligible['roa_pp'] = 100 * eligible.net_profit / eligible.assets
    eligible.groupby('company').roa_pp.agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index().to_csv(dest / 'annual_roa_summary.csv', index=False)
    quarter.groupby(['company', 'company_name']).agg(rows=('period', 'size'),
            first_period=('period', 'min'), last_period=('period', 'max')).reset_index().to_csv(dest / 'quarterly_coverage.csv', index=False)
    p = pd.read_csv(ROOT / 'results/annual/predictions.csv')
    p = p[p.scenario.eq('main_ifrs') & p.year.between(2018, 2024)].copy()
    p['error_pp'] = 100 * (p.prediction - p.actual)
    p['absolute_error_pp'] = p.error_pp.abs()
    p['squared_error_pp'] = p.error_pp ** 2
    for group, file in [('company', 'annual_errors_by_company.csv'), ('year', 'annual_errors_by_year.csv')]:
        out = p.groupby(['model', group]).agg(N=('actual', 'size'), MAE_pp=('absolute_error_pp', 'mean'),
                       MSE_pp=('squared_error_pp', 'mean'), bias_pp=('error_pp', 'mean')).reset_index()
        out['RMSE_pp'] = np.sqrt(out.pop('MSE_pp'))
        out.to_csv(dest / file, index=False)
    p.sort_values('absolute_error_pp', ascending=False).groupby('model', sort=False).head(10).to_csv(dest / 'annual_largest_errors.csv', index=False)
    gb = p[p.model.eq('GradientBoosting internal')]
    naive = p[p.model.eq('Naive last ROA')]
    pair = gb.merge(naive, on=['company', 'year'], suffixes=('_gb', '_naive'), validate='one_to_one')
    summary = {'scope': 'Original annual main_ifrs, 2018-2024; not a new model fit',
               'cases': len(pair), 'gb_better_cases': int((pair.absolute_error_pp_gb < pair.absolute_error_pp_naive).sum()),
               'gb_top10_absolute_error_share': float(gb.absolute_error_pp.nlargest(10).sum() / gb.absolute_error_pp.sum()),
               'gb_max_absolute_error_pp': float(gb.absolute_error_pp.max())}
    (dest / 'error_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Wrote descriptive tables to ' + str(dest.relative_to(ROOT)))


if __name__ == '__main__':
    main()
