"""Three pre-specified, separate modifications of the frozen annual ROA pilot.

Run with bundled Python. Missing ML libraries are read from the existing model
environment; bundled numpy/pandas take precedence. No source dataset is edited.
"""
from pathlib import Path
import sys, os, json, hashlib, platform, time

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'tmp/model_env_20261016/Lib/site-packages'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import sklearn, catboost, scipy
from retest_harmonized_models_20261016 import estimator, metrics, GRIDS, SEED, NUM, MACRO, CAT
from harmonize_dataset_20261016 import make_features

SRC = ROOT / 'outputs/dataset_harmonized_2026-10-16'
OUT = ROOT / 'outputs/annual_experiments_2026-10-19'
YEARS = list(range(2018, 2025))
NAIVE = 'Naive last ROA'
P0 = 'P0 | ML selector'
E1 = 'E1 | ML + naive selector'
E2 = 'E2 | delta selector'
E3 = 'E3 | loss selector'
HEADLINE = [NAIVE, 'GradientBoosting internal', P0, E1, E2, E3]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def save(name, frame):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def recs(frame):
    return json.loads(frame.to_json(orient='records', force_ascii=False, double_precision=15))

def specs():
    result = []
    for target in ['level', 'delta']:
        for family in GRIDS:
            for macro in [False, True]:
                name = family + (' + macro' if macro else ' internal')
                if target == 'delta':
                    name = 'Delta | ' + name
                result.append(dict(name=name, target=target, family=family, macro=macro, loss='original'))
    for family, losses in [('GradientBoosting', ['absolute_error', 'huber']), ('CatBoost', ['MAE'])]:
        for loss in losses:
            for macro in [False, True]:
                name = f'Loss | {family} {loss}' + (' + macro' if macro else ' internal')
                result.append(dict(name=name, target='level', family=family, macro=macro, loss=loss))
    return result

def make_model(spec, parameter):
    model = estimator(spec['family'], parameter, spec['macro'])
    if spec['loss'] != 'original':
        if spec['family'] == 'GradientBoosting':
            model.set_params(model__loss=spec['loss'], model__alpha=0.9)
        else:
            model.set_params(model__loss_function=spec['loss'])
    return model

def inner_years(data, outer):
    train = data[data.year < outer]
    return [int(y) for y in sorted(train.year.unique())
            if (train.year < y).sum() >= 20 and (train.year == y).sum() >= 3][-3:]

def run():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {p.name: sha(p) for p in SRC.iterdir() if p.is_file()}
    source_code = {n: sha(ROOT/n) for n in ['retest_harmonized_models_20261016.py', 'harmonize_dataset_20261016.py']}
    companies = pd.read_csv(SRC/'company_financials.csv')
    macro = pd.read_csv(SRC/'macro_by_year.csv')
    data = make_features(companies, macro)
    data = data[data.year <= 2024].copy()
    original = pd.read_csv(SRC/'predictions.csv')
    original = original[original.scenario.eq('main_ifrs') & original.year.isin(YEARS)].copy()
    old_sel = pd.read_csv(SRC/'selected_parameters.csv')
    old_sel = old_sel[old_sel.scenario.eq('main_ifrs') & old_sel.year.isin(YEARS)]
    old_folds = pd.read_csv(SRC/'folds.csv')
    old_folds = old_folds[old_folds.scenario.eq('main_ifrs') & old_folds.year.isin(YEARS)]
    checks = []
    def check(name, condition):
        checks.append(dict(test=name, passed=bool(condition)))
        assert condition, name

    check('unique_company_year', not data.duplicated(['company','year']).any())
    check('2025_completely_excluded', data.year.max() == 2024)
    check('same_test_keys', set(map(tuple, data[data.year.isin(YEARS)][['company','year']].values)) ==
          set(map(tuple, original[['company','year']].values)))
    check('same_test_count_88', len(data[data.year.isin(YEARS)]) == 88)
    check('past_features_only', (data.feature_year < data.year).all() and (data.macro_year < data.year).all())
    check('no_target_predictors', not set(NUM+MACRO+CAT) & {'actual','target_break','target_discontinued'})
    altered = companies.copy()
    altered.loc[altered.year.eq(2023), 'net_profit'] *= 9
    a = make_features(altered, macro)
    check('2023_target_perturbation_leaves_predictors_unchanged',
          data[data.year.eq(2023)][NUM+MACRO+CAT].equals(a[a.year.eq(2023)][NUM+MACRO+CAT]))
    check('row_order_invariance', make_features(companies, macro).equals(
          make_features(companies.sample(frac=1, random_state=7), macro)))

    plan = dict(date='2026-10-19', status='registered_before_fitting', scope='main_ifrs only',
        experiments={
            'E1': 'Original 8 level-model variants plus last-ROA baseline in the past-fold selector. No new model fits/features/losses.',
            'E2': 'The same 8 variants learn 100*(ROA_t-ROA_t-1), then add ROA_t-1. Original losses/features/grids. ML-only selector, no E1 combination.',
            'E3': 'Level target. Original 8 variants plus GB absolute_error/Huber(alpha=.9) and CatBoost MAE, each with/without macro. ML-only selector, no E1/E2 combination.'},
        primary_comparator=P0, additional_comparators=[NAIVE,'GradientBoosting internal'],
        primary_outputs=[E1,E2,E3], secondary_outputs='All family/loss variants; any best variant identified on the outer test is descriptive, not an independently selected winner.',
        sample='Same 88 cases, 14 companies, seven outer years 2018-2024; no deletions, clipping or added variables.',
        validation='Same expanding calendar-year folds, minimum 20 training/3 inner-validation observations, last three admissible prior years; pooled MAE selection.',
        selection='Within variant: minimum inner MAE then lower parameter. Between variants: minimum inner MAE then name. E1 exact ties prefer naive.',
        features_internal=NUM, features_macro=MACRO, categories=CAT, grids=GRIDS, seed=SEED,
        fitting='Original preprocessing fitted anew inside each training fold; original tree counts/depth grid/learning rates. No early stopping.',
        units='Predictions/targets stored as ratios; losses trained in percentage points; metrics in percentage points.',
        uncertainty='Descriptive paired cluster bootstrap of error differences by year and relationship group separately, 5000 replicates, not significance proof.',
        limitations=['Post-diagnostic exploratory experiments on previously inspected test years, not an untouched confirmatory test.',
            'Historical publication availability and revised source vintages are not reconstructed.',
            'Seven test years and related enterprises limit statistical independence.',
            'No improvement guaranteed; all planned outcomes, including negative ones, retained.'],
        versions={m.__name__:m.__version__ for m in [np,pd,sklearn,catboost,scipy]},
        python=platform.python_version(), source_hashes=frozen, original_code_hashes=source_code,
        experiment_code_sha256=sha(Path(__file__)),
        implementation_docs=['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html',
                             'https://catboost.ai/docs/en/concepts/loss-functions-regression'])
    write_json('experiment_protocol.json', plan)
    # Cache identical train-to-validation-year fits across overlapping inner folds.
    # No future labels influence a cached fit or preprocessing.
    cache = {}
    fit_log = []
    def forecast(spec, parameter, year):
        key = (spec['name'], parameter, int(year))
        if key not in cache:
            train = data[data.year < year]
            test = data[data.year.eq(year)]
            check(f'past_train_{len(cache)}', train.year.max() < year)
            model = make_model(spec, parameter)
            target = 100 * train.actual.to_numpy()
            if spec['target'] == 'delta':
                target = target - 100 * train.roa_lag.to_numpy()
            model.fit(train, target)
            pred_pp = model.predict(test)
            if spec['target'] == 'delta':
                pred_pp = pred_pp + 100 * test.roa_lag.to_numpy()
            pred = pred_pp / 100
            assert np.isfinite(pred).all()
            cache[key] = pred
            fit_log.append(dict(model=spec['name'],parameter=parameter,forecast_year=int(year),
                                train_n=len(train),test_n=len(test),max_train_year=int(train.year.max())))
        return cache[key]

    predictions, selection, cv_rows, folds = [], [], [], []
    specification = specs()
    level_names = [s['name'] for s in specification if s['target']=='level' and s['loss']=='original']
    delta_names = [s['name'] for s in specification if s['target']=='delta']
    loss_names = [s['name'] for s in specification if s['loss']!='original']
    for year in YEARS:
        train, test = data[data.year < year], data[data.year.eq(year)]
        iy = inner_years(data, year)
        old_fold = old_folds[old_folds.year.eq(year)].iloc[0]
        check(f'fold_sizes_{year}', len(train)==old_fold.train and len(test)==old_fold.test)
        check(f'inner_years_{year}', ','.join(map(str,iy)) == old_sel[old_sel.year.eq(year)].iloc[0].inner_years)
        folds.append(dict(year=year, train=len(train), test=len(test), max_train_year=int(train.year.max()), inner_years=','.join(map(str,iy))))
        choices, forecasts = {}, {NAIVE:test.roa_lag.to_numpy(), 'Mean last 3':test.mean3.to_numpy()}
        for spec in specification:
            candidates=[]
            for parameter in GRIDS[spec['family']]:
                errors=[]
                for valid_year in iy:
                    val=data[data.year.eq(valid_year)]
                    pred=forecast(spec, parameter, valid_year)
                    e=100*np.abs(pred-val.actual.to_numpy())
                    errors.extend(e.tolist())
                    cv_rows.append(dict(outer_year=year,inner_year=valid_year,model=spec['name'],parameter=parameter,
                                        valid_n=len(val),MAE_pp=float(e.mean())))
                candidates.append((float(np.mean(errors)),parameter))
            cv, parameter=min(candidates)
            choices[spec['name']]=(cv,parameter)
            forecasts[spec['name']]=forecast(spec,parameter,year)
            selection.append(dict(year=year,model=spec['name'],chosen_model=spec['name'],parameter=parameter,inner_MAE_pp=cv))
        naive_errors=[]
        for valid_year in iy:
            val=data[data.year.eq(valid_year)]
            e=100*np.abs(val.roa_lag.to_numpy()-val.actual.to_numpy())
            naive_errors.extend(e.tolist())
            cv_rows.append(dict(outer_year=year,inner_year=valid_year,model=NAIVE,parameter=None,valid_n=len(val),MAE_pp=float(e.mean())))
        choices[NAIVE]=(float(np.mean(naive_errors)),None)
        for label, names in [(P0,level_names),(E1,level_names+[NAIVE]),(E2,delta_names),(E3,level_names+loss_names)]:
            chosen=min(names,key=lambda n:(choices[n][0],0 if label==E1 and n==NAIVE else 1,n))
            forecasts[label]=forecasts[chosen].copy()
            selection.append(dict(year=year,model=label,chosen_model=chosen,parameter=choices[chosen][1],inner_MAE_pp=choices[chosen][0]))
        for name in level_names:
            old=old_sel[old_sel.year.eq(year)&old_sel.model.eq(name)].iloc[0]
            check(f'original_parameter_{year}_{name}', float(choices[name][1])==float(old.parameter))
            check(f'original_inner_score_{year}_{name}', np.isclose(choices[name][0],old.inner_MAE_pp,atol=1e-9,rtol=0))
        for name, pred in forecasts.items():
            for (_,row), value in zip(test.iterrows(),pred):
                predictions.append(dict(model=name,company=row.company,year=year,group=row.group,actual=float(row.actual),
                                        prediction=float(value),target_break=bool(row.target_break),target_discontinued=bool(row.target_discontinued)))
        print(f'{year}: ' + '; '.join(f'{x}: {next(z["chosen_model"] for z in selection if z["year"]==year and z["model"]==x)}' for x in [E1,E2,E3]),flush=True)

    p=pd.DataFrame(predictions)
    check('no_duplicate_predictions', not p.duplicated(['model','company','year']).any())
    check('all_28_methods_same_88_cases', p.groupby('model').size().eq(88).all() and p.model.nunique()==28)
    control = p[p.model.isin(level_names+[NAIVE,'Mean last 3',P0])].copy()
    control['model']=control.model.replace({P0:'Selected on past folds'})
    merged=control.merge(original,on=['model','company','year'],suffixes=('_new','_old'),validate='one_to_one')
    check('original_968_predictions_reproduced', len(merged)==968 and np.allclose(merged.prediction_new,merged.prediction_old,atol=1e-10,rtol=0))
    check('original_targets_unchanged', np.allclose(merged.actual_new,merged.actual_old,atol=1e-14,rtol=0))
    m=metrics(p,by=['model'])
    naive_mae=float(m.loc[m.model.eq(NAIVE),'MAE_pp'].iloc[0])
    old_mae=float(m.loc[m.model.eq(P0),'MAE_pp'].iloc[0])
    m['delta_MAE_vs_original_selector_pp']=m.MAE_pp-old_mae
    m['improvement_vs_naive_pct']=100*(1-m.MAE_pp/naive_mae)
    m['improvement_vs_original_selector_pct']=100*(1-m.MAE_pp/old_mae)
    by_year=metrics(p,by=['model','year'])
    by_company=metrics(p,by=['model','company'])
    p['ae_pp']=100*(p.prediction-p.actual).abs()
    p['se_pp2']=(100*(p.prediction-p.actual))**2
    wide=p.pivot(index=['company','year','group'],columns='model',values='ae_pp')
    paired=[]
    boot=[]
    for name in p.model.unique():
        for comparator in [NAIVE,P0,'GradientBoosting internal']:
            diff=wide[name]-wide[comparator]
            annual=diff.groupby(level='year').mean()
            co=diff.groupby(level='company').mean()
            paired.append(dict(model=name,comparator=comparator,delta_MAE_pp=float(diff.mean()),
                cases_better=int((diff < -1e-10).sum()),cases_worse=int((diff > 1e-10).sum()),
                cases_equal=int((diff.abs() <= 1e-10).sum()),years_better=int((annual < -1e-10).sum()),
                companies_better=int((co < -1e-10).sum())))
            if name not in [E1,E2,E3]: continue
            for cluster in ['year','group']:
                groups=[v.to_numpy() for _,v in diff.groupby(level=cluster)]
                rng=np.random.default_rng(SEED)
                sims=np.array([np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(5000)])
                boot.append(dict(model=name,comparator=comparator,cluster=cluster,n_clusters=len(groups),
                                 delta_MAE_pp=float(diff.mean()),lo_pp=float(np.quantile(sims,.025)),hi_pp=float(np.quantile(sims,.975))))
    check('frozen_source_files_unchanged', frozen=={f.name:sha(f) for f in SRC.iterdir() if f.is_file()})
    check('original_code_unchanged', all(sha(ROOT/n)==h for n,h in source_code.items()))
    cv=pd.DataFrame(cv_rows)
    check('inner_year_strictly_before_outer', cv.inner_year.lt(cv.outer_year).all())
    # Independently recompute saved baseline headline metrics.
    old_metrics=pd.read_csv(SRC/'model_metrics.csv')
    old_metrics=old_metrics[old_metrics.scenario.eq('main_ifrs')]
    for name in level_names+[NAIVE,'Mean last 3',P0]:
        old_name='Selected on past folds' if name==P0 else name
        x=m[m.model.eq(name)].iloc[0]; y=old_metrics[old_metrics.model.eq(old_name)].iloc[0]
        check('original_metric_'+name,np.allclose([x.MAE_pp,x.RMSE_pp],[y.MAE_pp,y.RMSE_pp],atol=1e-10,rtol=0))
    save('predictions.csv',p)
    save('metrics.csv',m.sort_values('MAE_pp'))
    save('headline_metrics.csv',m.set_index('model').loc[HEADLINE].reset_index())
    save('by_year.csv',by_year)
    save('by_company.csv',by_company)
    save('selected_models.csv',pd.DataFrame(selection))
    save('inner_cv.csv',cv)
    save('folds.csv',pd.DataFrame(folds))
    save('fit_log.csv',pd.DataFrame(fit_log))
    save('paired_comparisons.csv',pd.DataFrame(paired))
    save('paired_cluster_bootstrap.csv',pd.DataFrame(boot))
    save('checks.csv',pd.DataFrame(checks))
    write_json('results.json',dict(headline=recs(m.set_index('model').loc[HEADLINE].reset_index()),
        metrics=recs(m),by_year=recs(by_year),by_company=recs(by_company),paired=paired,bootstrap=boot,
        selections=recs(pd.DataFrame(selection)),predictions=recs(p)))
    write_json('run_complete.json',dict(status='complete',checks_passed=len(checks),unique_fits=len(cache),
        seconds=round(time.time()-started,2),test_cases=88,companies=14,test_years=YEARS,
        methods=28,original_max_abs_prediction_difference=float((merged.prediction_new-merged.prediction_old).abs().max()),
        original_files_unchanged=True,experiment_code_sha256=sha(Path(__file__))))
    print('\nHEADLINE\n'+m.set_index('model').loc[HEADLINE].round(5).to_string(),flush=True)
    print('\nALL MODELS\n'+m[['model','MAE_pp','RMSE_pp']].sort_values('MAE_pp').round(4).to_string(index=False),flush=True)
    print(f'COMPLETE: {len(checks)} checks; {len(cache)} fits; {time.time()-started:.1f} seconds',flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
