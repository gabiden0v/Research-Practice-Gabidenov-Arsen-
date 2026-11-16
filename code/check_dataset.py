"""Check the packaged data and feature arithmetic, without changing source tables."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/original'))
from harmonize_dataset_20261016 import make_features


def main():
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})

    annual = pd.read_csv(ROOT / 'data/annual/company_financials.csv')
    macro = pd.read_csv(ROOT / 'data/annual/macro_by_year.csv')
    quarterly = pd.read_csv(ROOT / 'data/quarterly/company_quarterly.csv')
    check('annual: 196 rows, 15 firms', len(annual) == 196 and annual.company.nunique() == 15)
    check('annual: unique company-year', not annual.duplicated(['company', 'year']).any())
    eligible = annual[annual.main_eligible]
    check('annual: 165 eligible rows, 14 firms', len(eligible) == 165 and eligible.company.nunique() == 14)
    features = make_features(annual, macro)
    frozen = pd.read_csv(ROOT / 'data/annual/model_features.csv')
    check('annual: 151 feature rows', len(features) == len(frozen) == 151)
    a = features.sort_values(['company', 'year']).reset_index(drop=True)
    b = frozen.sort_values(['company', 'year']).reset_index(drop=True)
    check('annual: feature keys match', a[['company', 'year']].equals(b[['company', 'year']]))
    numeric = list(a.select_dtypes(include='number').columns)
    check('annual: all numeric features reproduced', np.allclose(a[numeric], b[numeric], atol=1e-10, rtol=1e-10, equal_nan=True))
    check('annual: features precede target year', (a.feature_year < a.year).all() and (a.macro_year < a.year).all())
    target = eligible[['company', 'year', 'net_profit', 'assets']].merge(a[['company', 'year', 'actual']], on=['company', 'year'], validate='one_to_one')
    check('annual: target is profit / end assets', np.allclose(target.actual, target.net_profit / target.assets, atol=1e-12))
    check('quarterly: 72 rows, 3 firms', len(quarterly) == 72 and quarterly.company.nunique() == 3)
    check('quarterly: unique company-period', not quarterly.duplicated(['company', 'period']).any())
    check('quarterly: ROA arithmetic', np.allclose(quarterly.roa, quarterly.profit_quarter_m / quarterly.assets_m, atol=1e-12))
    q = quarterly.sort_values(['company', 'year', 'quarter']).copy()
    expected_profit = q.groupby(['company', 'year']).profit_ytd_m.diff()
    expected_profit = expected_profit.where(q.quarter.ne(1), q.profit_ytd_m)
    check('quarterly: YTD difference arithmetic', np.allclose(expected_profit, q.profit_quarter_m, atol=1e-8))
    qf = pd.read_csv(ROOT / 'data/quarterly/features_main_lag2.csv')
    check('quarterly: 60 feature rows', len(qf) == 60)
    check('quarterly: feature delay at least two quarters', (qf.feature_t <= qf.t - 2).all())
    for folder, keys in [('annual', ['scenario', 'model', 'company', 'year']),
                         ('annual_extensions', ['model', 'company', 'year']),
                         ('quarterly', ['scenario', 'model', 'company', 'period'])]:
        p = pd.read_csv(ROOT / 'results' / folder / 'predictions.csv')
        check(folder + ': unique prediction keys', not p.duplicated(keys).any())
        check(folder + ': finite predictions and targets', np.isfinite(p[['actual', 'prediction']]).all().all())
        if folder == 'annual':
            main = p[p.scenario.eq('main_ifrs') & p.year.between(2018, 2024)]
            check('annual: 88 primary cases per model', main.groupby('model').size().eq(88).all())
        elif folder == 'quarterly':
            main = p[p.scenario.eq('main_lag2')]
            check('quarterly: 24 primary cases per model', main.groupby('model').size().eq(24).all())
            target_ord = pd.PeriodIndex(main.period, freq='Q').asi8
            label_ord = pd.PeriodIndex(main.last_label_period, freq='Q').asi8
            check('quarterly: training labels end by target minus two', (label_ord <= target_ord - 2).all())
    dest = ROOT / 'recomputed'
    dest.mkdir(exist_ok=True)
    (dest / 'dataset_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    for item in checks:
        print(('PASS ' if item['passed'] else 'FAIL ') + item['check'])
    if not all(x['passed'] for x in checks):
        raise SystemExit('Dataset checks failed; inspect recomputed/dataset_checks.json')


if __name__ == '__main__':
    main()
