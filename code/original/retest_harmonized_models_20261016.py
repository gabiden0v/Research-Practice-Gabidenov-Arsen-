"""Library-based, nested expanding-year benchmark on versioned harmonised data.

All selection and preprocessing use past folds only. Revised source vintages mean
this remains a retrospective backtest, NOT a historical real-time simulation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json,sys,hashlib,platform
import numpy as np
import pandas as pd
import sklearn, catboost, scipy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from harmonize_dataset_20261016 import OUT,ROOT,make_features,sha

NUM=['roa_lag','equity_lag','asset_growth_lag','revenue_growth_lag']
MACRO=['gdp_growth','inflation','fx_change','brent_change']
CAT=['company','sector']
GRIDS={'Ridge':[1.0,10.0,100.0], 'RandomForest':[2,4],
       'GradientBoosting':[1,2], 'CatBoost':[2,4]}
SEED=20261016

def estimator(family,parameter,macro):
    numeric=NUM+(MACRO if macro else [])
    prep=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),numeric),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],sparse_threshold=0)
    if family=='Ridge':m=Ridge(alpha=parameter)
    elif family=='RandomForest':m=RandomForestRegressor(n_estimators=150,max_depth=parameter,min_samples_leaf=5,max_features=0.8,random_state=SEED,n_jobs=2)
    elif family=='GradientBoosting':m=GradientBoostingRegressor(n_estimators=150,learning_rate=.03,max_depth=parameter,min_samples_leaf=5,loss='squared_error',random_state=SEED)
    else:m=CatBoostRegressor(iterations=200,depth=parameter,learning_rate=.03,l2_leaf_reg=5,loss_function='RMSE',random_seed=SEED,thread_count=2,verbose=False,allow_writing_files=False)
    return Pipeline([('prepare',prep),('model',m)])

def metrics(frame,by=['scenario','model']):
    out=[]
    for keys,g in frame.groupby(by,sort=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        err=(g.prediction-g.actual).to_numpy()*100
        actual=g.actual.to_numpy()*100
        den=np.sum((actual-actual.mean())**2)
        out.append(dict(zip(by,keys),N=len(g),MAE_pp=float(np.mean(np.abs(err))),RMSE_pp=float(np.sqrt(np.mean(err**2))),
            bias_pp=float(err.mean()),R2=float(1-np.sum(err**2)/den) if den>0 else None))
    return pd.DataFrame(out)

def run(data,scenario,lag=1,years=range(2018,2026)):
    predictions=[];folds=[];selections=[];cv_rows=[]
    specs=[(f,macro) for f in GRIDS for macro in [False,True]]
    for year in years:
        train=data[data.year<=year-lag];test=data[data.year==year]
        if len(train)<20 or test.empty:continue
        assert train.year.max()<year
        assert test.feature_year.max()<=year-lag
        folds.append(dict(scenario=scenario,year=year,train=len(train),test=len(test),max_train_year=int(train.year.max()),lag=lag,
            phase='exploratory_2025' if year==2025 else 'retrospective_test'))
        # Tune on at most 3 earlier calendar years, with the same publication-delay gap.
        inner=[]
        for iy in sorted(train.year.unique()):
            itr=train[train.year<=iy-lag];ite=train[train.year==iy]
            if len(itr)>=20 and len(ite)>=3:inner.append((iy,itr,ite))
        inner=inner[-3:]
        if not inner:raise RuntimeError(f'Insufficient past validation folds for {scenario} {year}')
        best_predictions={};selection_candidates=[]
        for family,macro in specs:
            model_name=family+(' + macro' if macro else ' internal')
            candidates=[]
            for parameter in GRIDS[family]:
                errors=[]
                for iy,itr,ite in inner:
                    model=estimator(family,parameter,macro)
                    model.fit(itr,itr.actual*100)
                    pred=model.predict(ite)
                    e=np.abs(pred-ite.actual.to_numpy()*100)
                    errors.extend(e.tolist())
                    cv_rows.append(dict(scenario=scenario,outer_year=year,inner_year=int(iy),model=model_name,parameter=parameter,
                        train_n=len(itr),valid_n=len(ite),max_train_year=int(itr.year.max()),MAE_pp=float(e.mean())))
                candidates.append((float(np.mean(errors)),parameter))
            cv,parameter=min(candidates,key=lambda x:(x[0],x[1]))
            model=estimator(family,parameter,macro)
            model.fit(train,train.actual*100)
            p=model.predict(test)/100
            best_predictions[model_name]=p
            selection_candidates.append((cv,model_name,parameter))
            selections.append(dict(scenario=scenario,year=year,model=model_name,parameter=parameter,inner_MAE_pp=cv,
                inner_years=','.join(str(int(x[0])) for x in inner)))
        _,chosen,parameter=min(selection_candidates,key=lambda x:(x[0],x[1]))
        best_predictions['Naive last ROA']=test.roa_lag.to_numpy()
        best_predictions['Mean last 3']=test.mean3.to_numpy()
        best_predictions['Selected on past folds']=best_predictions[chosen].copy()
        selections.append(dict(scenario=scenario,year=year,model='Selected on past folds',parameter=str(parameter),
            inner_MAE_pp=min(selection_candidates)[0],selected_model=chosen,
            inner_years=','.join(str(int(x[0])) for x in inner)))
        for model,pred in best_predictions.items():
            for (_,row),p in zip(test.iterrows(),pred):
                predictions.append(dict(scenario=scenario,model=model,company=row.company,year=int(row.year),group=row.group,
                    actual=float(row.actual),prediction=float(p),target_break=bool(row.target_break),target_discontinued=bool(row.target_discontinued)))
        print(f'{scenario} {year}: train={len(train)}, test={len(test)}, past-selected={chosen}',flush=True)
    return pd.DataFrame(predictions),pd.DataFrame(folds),pd.DataFrame(selections),pd.DataFrame(cv_rows)

def main():
    companies=pd.read_csv(OUT/'company_financials.csv')
    macro=pd.read_csv(OUT/'macro_by_year.csv')
    main_data=make_features(companies,macro)
    broad_data=make_features(companies,macro,allow_scope_uncertain=True)
    delayed=make_features(companies,macro,lag=2)
    no_parents=make_features(companies[~companies.company.isin(['НК КазМунайГаз','Казахтелеком'])],macro)
    protocol=dict(target='ROA = total group/company after-tax profit / year-end total assets; ratio units',
        main_scenario='main_ifrs',test_years=list(range(2018,2025)),exploratory_year=2025,
        task='One-step reporting-year ROA prediction conditional on previous annual statements, retrospective source-selected vintages',
        features_internal=NUM,features_macro=MACRO,categorical_features=CAT,
        primary_metric='MAE in percentage points; RMSE and pooled R2 secondary',
        validation='Expanding outer calendar-year folds; hyperparameters selected by pooled MAE in last 3 admissible past calendar-year inner folds; preprocessing fit separately inside every fold.',
        grids=GRIDS,implementations='scikit-learn and CatBoost official libraries, not previous custom NumPy implementations',
        fixed_parameters={'RF':{'n_estimators':150,'min_samples_leaf':5,'max_features':.8},
            'GB':{'n_estimators':150,'learning_rate':.03,'min_samples_leaf':5},
            'CatBoost':{'iterations':200,'learning_rate':.03,'l2_leaf_reg':5},'random_seed':SEED},
        early_stopping='None; neither outer test labels nor 2025 labels used for tuning',
        macro_ablation='Identical sample and folds; independently tuned family with and without four macro features',
        missing_values='Target and lagged ROA required; other numeric gaps median-imputed with missingness indicators, fitted only on training data. Source gaps never overwritten.',
        scenarios={'main_ifrs':'Excludes TCO, incomplete Ozen 2012, uncertain Ozen scope through 2022, standalone MMG 2025. Retains economic shocks and discontinued profits in total-profit ROA.',
            'broad_scope_sensitivity':'Adds Ozen full years with unresolved scope; NOT promoted to primary.',
            'no_parent_overlap':'Removes KMG and Kazakhtelecom parents from training/test; related subsidiary dependencies may remain.',
            'lag2_delay_sensitivity':'Company and macro features dated y-2, training outcomes no later than y-2; conservative delay check, not historical vintage reconstruction.'},
        limitations=['Sources include retrospective restatements and 2026 macro revisions; actual publication availability remains unverified.',
            '2025 was already inspected in earlier work, has at most four comparable companies, and is NOT an untouched holdout.',
            'Only seven outer test years, few related company groups; rankings and cluster bootstrap intervals are exploratory.',
            'Headline differences from previous results mix protocol, sample and implementation changes; not a causal effect of cleaning.',
            'Total-profit ROA deliberately includes one-off/discontinued results; not a normalised operating-return forecast.',
            'No claims of universal best model or neural-network superiority; no prediction intervals are calibrated in this run.'],
        versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'catboost':catboost.__version__,'scipy':scipy.__version__},
        input_hashes={f:sha(OUT/f) for f in ['company_financials.csv','macro_by_year.csv','source_registry.json']})
    (OUT/'model_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    all_p=[];all_f=[];all_s=[];all_cv=[]
    for name,d,lag in [('main_ifrs',main_data,1),('broad_scope_sensitivity',broad_data,1),('no_parent_overlap',no_parents,1),('lag2_delay_sensitivity',delayed,2)]:
        p,f,s,cv=run(d,name,lag)
        assert p.groupby(['company','year']).model.nunique().eq(11).all()
        assert not p.duplicated(['company','year','model']).any()
        assert np.isfinite(p.prediction).all()
        all_p.append(p);all_f.append(f);all_s.append(s);all_cv.append(cv)
        pd.concat(all_p).to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    p=pd.concat(all_p,ignore_index=True);test=p[p.year.between(2018,2024)]
    m=metrics(test)
    naive=m[m.model.eq('Naive last ROA')].set_index('scenario').MAE_pp
    m['MAE_improvement_vs_naive_pct']=100*(1-m.MAE_pp/m.scenario.map(naive))
    m=m.sort_values(['scenario','MAE_pp'])
    m.to_csv(OUT/'model_metrics.csv',index=False,encoding='utf-8-sig')
    metrics(p[p.year.eq(2025)]).to_csv(OUT/'exploratory_2025.csv',index=False,encoding='utf-8-sig')
    metrics(test,['scenario','model','company']).to_csv(OUT/'metrics_by_company.csv',index=False,encoding='utf-8-sig')
    metrics(test,['scenario','model','year']).to_csv(OUT/'metrics_by_year.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_f).to_csv(OUT/'folds.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_s).to_csv(OUT/'selected_parameters.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_cv).to_csv(OUT/'inner_cv_results.csv',index=False,encoding='utf-8-sig')
    ab=[]
    for scenario,g in m.groupby('scenario'):
        for fam in GRIDS:
            a=g[g.model.eq(fam+' internal')].iloc[0];b=g[g.model.eq(fam+' + macro')].iloc[0]
            assert a.N==b.N
            ab.append(dict(scenario=scenario,family=fam,N=int(a.N),MAE_internal_pp=a.MAE_pp,MAE_macro_pp=b.MAE_pp,
                MAE_macro_minus_internal_pp=b.MAE_pp-a.MAE_pp,RMSE_macro_minus_internal_pp=b.RMSE_pp-a.RMSE_pp))
    pd.DataFrame(ab).to_csv(OUT/'macro_ablation.csv',index=False,encoding='utf-8-sig')
    # Paired cluster resampling: descriptive, not independent-company significance.
    b=[];loss=test[test.scenario.eq('main_ifrs')].copy()
    loss['ae']=(loss.prediction-loss.actual).abs()*100
    wide=loss.pivot(index=['company','year','group'],columns='model',values='ae')
    for model in wide.columns.drop('Naive last ROA'):
        diff=wide[model]-wide['Naive last ROA']
        for cluster in ['year','group']:
            groups=[z.to_numpy() for _,z in diff.groupby(level=cluster)]
            rng=np.random.default_rng(SEED)
            sims=[np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(2000)]
            b.append(dict(model=model,cluster=cluster,n_clusters=len(groups),delta_MAE_pp=float(diff.mean()),
                lo_pp=float(np.quantile(sims,.025)),hi_pp=float(np.quantile(sims,.975))))
    pd.DataFrame(b).to_csv(OUT/'paired_cluster_bootstrap.csv',index=False,encoding='utf-8-sig')
    # Numeric/methodological invariants.
    checks=[]
    def check(name,value):
        checks.append(dict(test=name,passed=bool(value)));assert value,name
    check('unique_input_keys',not companies.duplicated(['company','year']).any())
    check('row_order_invariance',main_data.equals(make_features(companies.sample(frac=1,random_state=7),macro)))
    future=companies.copy();future.loc[future.year.ge(2024),['sales_revenue','reported_revenue','net_profit','assets','equity']]*=7
    altered=make_features(future,macro)
    check('future_rows_do_not_change_past_features',main_data[main_data.year.le(2023)].equals(altered[altered.year.le(2023)]))
    gap=companies[~(companies.company.eq('Air Astana') & companies.year.eq(2017))]
    gf=make_features(gap,macro)
    check('missing_year_not_bridged',gf[gf.company.eq('Air Astana') & gf.year.eq(2018)].empty)
    check('target_not_in_predictor_list',not set(NUM+MACRO+CAT)&{'actual','target_break','target_discontinued'})
    target_changed=companies.copy();target_changed.loc[target_changed.year.eq(2023),'net_profit']*=9
    tf=make_features(target_changed,macro)
    check('target_year_outcome_not_in_predictors',main_data[main_data.year.eq(2023)][NUM+MACRO+CAT].equals(tf[tf.year.eq(2023)][NUM+MACRO+CAT]))
    check('main_excludes_special_basis',not main_data.company.eq('Тенгизшевройл').any())
    check('2025_not_in_headline_tests',not test.year.eq(2025).any())
    cv=pd.concat(all_cv);check('inner_validation_before_outer',cv.inner_year.lt(cv.outer_year).all())
    check('inner_training_before_validation',cv.max_train_year.lt(cv.inner_year).all())
    for k,v in protocol['input_hashes'].items():check('immutable_model_input_'+k,sha(OUT/k)==v)
    old=json.loads((OUT/'preparation_summary.json').read_text(encoding='utf-8'))['old_sha256']
    check('previous_deliverables_unchanged',all(sha(ROOT/k)==v for k,v in old.items()))
    pd.DataFrame(checks).to_csv(OUT/'model_checks.csv',index=False,encoding='utf-8-sig')
    (OUT/'run_complete.json').write_text(json.dumps({'checks_passed':len(checks),'predictions':len(p),
        'main_test_observations':int(len(test[test.scenario.eq('main_ifrs')])/11),'models_per_case':11,
        'code_sha256':sha(Path(__file__))},ensure_ascii=False,indent=2),encoding='utf-8')
    print(m[m.scenario.eq('main_ifrs')].to_string(index=False),flush=True)
    print('ALL CHECKS PASSED',len(checks),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
