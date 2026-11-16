"""Quarterly ROA pilot. Bundled Python; source annual files are read-only.

Run: bundled python -X utf8 run_quarterly_pilot_20261019.py
Source values were manually checked against the statement pages saved under tmp.
This is a retrospective research pilot, NOT a point-in-time trading/credit system.
"""
from pathlib import Path
import sys,os,json,hashlib,platform
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT/'tmp/model_env_20261016/Lib/site-packages'))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import sklearn,catboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor

BASE=ROOT/'tmp/quarterly_pilot_20261019'
OUT=ROOT/'outputs/quarterly_pilot_2026-10-19';OUT.mkdir(parents=True,exist_ok=True)
ANNUAL=ROOT/'outputs/dataset_harmonized_2026-10-16'
SEED=20261019
NAMES={'KCELL':'Kcell','KEGOC':'KEGOC','KZTO':'КазТрансОйл'}
SECTORS={'KCELL':'Мобильная связь','KEGOC':'Электроэнергетика','KZTO':'Транспортировка нефти'}
NAIVE='Seasonal naive';LAST='Last available ROA';GB='GB MAE internal';CB='CatBoost MAE + macro';CB0='CatBoost MAE internal (control)'
MODELS=[NAIVE,GB,CB,LAST,CB0]
checks=[]
def check(name,ok):
 checks.append({'check':name,'passed':bool(ok)})
 assert ok,name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def recs(df):return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=15))
def js(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save(name,df):df.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def build_data():
 raw=pd.read_csv(BASE/'reviewed_interim.csv').sort_values(['company','year','quarter']).reset_index(drop=True)
 save('reviewed_source_inputs.csv',raw)
 registry=json.loads((BASE/'source_registry.json').read_text(encoding='utf-8'))
 lookup={(r['company'],r['year'],r['quarter']):r for r in registry}
 check('72_unique_report_periods',len(raw)==72 and not raw.duplicated(['company','year','quarter']).any())
 raw['period']=raw.year.astype(str)+'Q'+raw.quarter.astype(str)
 raw['source_id']=raw.company+'_'+raw.period
 for field in ['assets','revenue_ytd','profit_ytd','profit_direct_quarter']:
  raw[field+'_m']=raw[field]*raw.unit.map({'thousand_KZT':.001,'million_KZT':1.})
 raw['profit_quarter_m']=raw.groupby(['company','year']).profit_ytd_m.diff().where(raw.quarter.ne(1),raw.profit_ytd_m)
 raw['revenue_quarter_m']=raw.groupby(['company','year']).revenue_ytd_m.diff().where(raw.quarter.ne(1),raw.revenue_ytd_m)
 raw['direct_difference_m']=raw.profit_quarter_m-raw.profit_direct_quarter_m
 raw['direct_discrepancy']=raw.direct_difference_m.abs().gt(1.01) & raw.profit_direct_quarter_m.notna()
 raw['revenue_basis_break']=raw.company.eq('KEGOC')&raw.year.eq(2021)&raw.quarter.eq(4)
 raw.loc[raw.revenue_basis_break,'revenue_quarter_m']=np.nan
 raw['roa']=raw.profit_quarter_m/raw.assets_m
 raw['profit_reported_basis_m']=raw.profit_direct_quarter_m.fillna(raw.profit_quarter_m)
 raw['reported_roa']=raw.profit_reported_basis_m/raw.assets_m
 raw['sector']=raw.company.map(SECTORS);raw['company_name']=raw.company.map(NAMES)
 notes=[];sources=[]
 for row in raw.itertuples():
  a=dict(lookup[(row.company,row.year,row.quarter)])
  check('source_hash_'+row.source_id,sha(ROOT/a['local_path'])==a['sha256'])
  check('statement_pages_'+row.source_id,max(row.asset_page,row.income_page)<=a['pages'])
  flags=[]
  if row.direct_discrepancy:flags.append(f'Разность YTD отличается от опубликованного квартала на {row.direct_difference_m:.3f} млн KZT; смешение версий/периодов требует сверки')
  if row.revenue_basis_break:flags.append('Изменение периметра продолжающейся деятельности: квартальная выручка не рассчитывается')
  if row.quarter==4:flags.append('Q4 = год − 9М; включает возможные годовые корректировки')
  if row.company=='KZTO' and row.year==2024 and row.quarter==4:flags.append('Использована текущая колонка отчёта 2024; в отчёте 2025 сравнительные суммы пересмотрены')
  if row.company=='KEGOC' and row.year==2021:flags.append('Прибыль общая, включая прекращённую деятельность; в конце года переоценка активов')
  notes.append('; '.join(flags))
  a.update(source_id=row.source_id,asset_page=int(row.asset_page),income_page=int(row.income_page),source_unit=row.unit,
    assets_raw=row.assets,revenue_ytd_raw=row.revenue_ytd,profit_ytd_raw=row.profit_ytd,
    direct_profit_raw=None if pd.isna(row.profit_direct_quarter) else row.profit_direct_quarter,
    review='54 interim records: statement image review; annual records: compared with previously checked annual panel and current downloaded report, current-year figures preferred')
  sources.append(a)
 raw['notes']=notes
 check('complete_24_quarters_each',raw.groupby('company').size().eq(24).all())
 check('positive_assets_finite_targets',raw.assets_m.gt(0).all() and np.isfinite(raw.roa).all())
 check('known_direct_discrepancies',set(raw.loc[raw.direct_discrepancy,'source_id'])=={'KCELL_2020Q3','KCELL_2024Q2'})
 annual=raw.groupby(['company','year']).agg(profit_quarters_sum_m=('profit_quarter_m','sum'),reported_quarters_sum_m=('profit_reported_basis_m','sum')).reset_index()
 fy=raw[raw.quarter.eq(4)][['company','year','profit_ytd_m','assets_m']]
 annual=annual.merge(fy,on=['company','year'])
 check('annual_profit_reconciliation',np.allclose(annual.profit_quarters_sum_m,annual.profit_ytd_m,atol=1e-8))
 frozen=pd.read_csv(ANNUAL/'company_financials.csv');frozen['company']=frozen.company.map({'Кселл':'KCELL','КазТрансОйл':'KZTO','KEGOC':'KEGOC'})
 frozen=frozen.dropna(subset=['company']);frozen['annual_frozen_profit_m']=frozen.net_profit/1000;frozen['annual_frozen_assets_m']=frozen.assets/1000
 annual=annual.merge(frozen[['company','year','annual_frozen_profit_m','annual_frozen_assets_m']],on=['company','year'])
 annual['profit_difference_to_frozen_m']=annual.profit_ytd_m-annual.annual_frozen_profit_m
 annual['asset_difference_to_frozen_m']=annual.assets_m-annual.annual_frozen_assets_m
 annual['note']='YTD-difference reconciliation is an arithmetic identity, not independent proof of correctness.'
 macro=pd.read_csv(BASE/'macro_quarterly.csv')
 check('macro_complete_36_quarters',len(macro)==36 and macro.usd_kzt_end.notna().all() and macro.brent_usd_bbl_avg.notna().all())
 check('brent_enough_observations',macro.brent_observations.between(55,70).all())
 save('company_quarterly.csv',raw);save('macro_quarterly.csv',macro);save('annual_reconciliation.csv',annual);js('source_registry.json',sources)
 return raw,macro,annual,sources

def features(raw,macro,delay=2,reported=False):
 d=raw.sort_values(['company','year','quarter']).copy();d['t']=pd.PeriodIndex(d.period,freq='Q').asi8
 d['target']=d['reported_roa' if reported else 'roa'];g=d.groupby('company')
 for lag in [delay,delay+1,4]:d[f'roa_lag{lag}']=g.target.shift(lag)
 d['asset_growth']=g.assets_m.shift(delay)/g.assets_m.shift(delay+1)-1
 d['feature_t']=d.t-delay;d['quarter_cat']=d.quarter.astype(str)
 m=macro.copy();m['feature_t']=pd.PeriodIndex(m.period,freq='Q').asi8
 d=d.merge(m[['feature_t','fx_change_qoq','brent_change_qoq']],on='feature_t',how='left')
 nums=[f'roa_lag{lag}' for lag in [delay,delay+1,4]]+['asset_growth']
 d=d[d.roa_lag4.notna()].copy()
 return d,nums

def model(name,nums):
 macro=name==CB;cols=nums+(['fx_change_qoq','brent_change_qoq'] if macro else [])
 prep=ColumnTransformer([('num',SimpleImputer(strategy='median'),cols),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['company','quarter_cat'])])
 if name==GB:m=GradientBoostingRegressor(loss='absolute_error',n_estimators=150,learning_rate=.03,max_depth=2,min_samples_leaf=5,random_state=SEED)
 else:m=CatBoostRegressor(loss_function='MAE',iterations=200,learning_rate=.03,depth=2,l2_leaf_reg=5,random_seed=SEED,thread_count=2,verbose=False,allow_writing_files=False)
 return Pipeline([('prep',prep),('model',m)])

def run_scenario(raw,macro,scenario,delay,reported):
 d,nums=features(raw,macro,delay,reported);pred=[];folds=[]
 for period in pd.period_range('2024Q1','2025Q4',freq='Q'):
  t=period.ordinal;train=d[d.t<=t-delay].copy();test=d[d.t.eq(t)].copy()
  check(f'past_labels_{scenario}_{period}',train.t.max()<=t-delay and len(train)>=30)
  check(f'3_test_companies_{scenario}_{period}',len(test)==3)
  check(f'past_features_{scenario}_{period}',test.feature_t.lt(t).all())
  for name in MODELS:
   if name==NAIVE:p=test.roa_lag4.to_numpy()
   elif name==LAST:p=test[f'roa_lag{delay}'].to_numpy()
   else:
    est=model(name,nums);est.fit(train,train.target*100);p=est.predict(test)/100
   for (_,r),v in zip(test.iterrows(),p):
    pred.append(dict(scenario=scenario,model=name,company=r.company,company_name=r.company_name,year=int(r.year),quarter=int(r.quarter),period=r.period,actual=r.target,prediction=float(v),train_n=len(train),last_label_period=str(pd.Period(ordinal=int(train.t.max()),freq='Q')),feature_period=str(pd.Period(ordinal=int(r.feature_t),freq='Q')),data_flag=bool(r.direct_discrepancy)))
  folds.append(dict(scenario=scenario,period=str(period),train_n=len(train),train_last=str(pd.Period(ordinal=int(train.t.max()),freq='Q')),test_n=len(test)))
 p=pd.DataFrame(pred);p['error_pp']=(p.prediction-p.actual)*100;p['ae_pp']=p.error_pp.abs();p['se_pp']=p.error_pp**2
 print(scenario,'complete',flush=True)
 return p,pd.DataFrame(folds),d,nums

def metrics(df,groups):
 return df.groupby(groups,dropna=False).agg(N=('ae_pp','size'),MAE_pp=('ae_pp','mean'),MSE_pp=('se_pp','mean'),bias_pp=('error_pp','mean')).reset_index().assign(RMSE_pp=lambda x:np.sqrt(x.MSE_pp)).drop(columns='MSE_pp')

def main():
 frozen={p.name:sha(p) for p in ANNUAL.iterdir() if p.is_file()}
 raw,macro,annual,sources=build_data()
 protocol=dict(status='fixed_before_model_fitting',date='2026-10-19',coverage='3 companies, 2020Q1–2025Q4, 72 company-quarter rows',
  primary_target='Non-annualised quarterly ROA = (profit YTD_t - profit YTD_previous same year) / assets at quarter end; Q1 uses its YTD. Total profit after tax, not comprehensive income.',
  primary_test='2024Q1–2025Q4; 8 expanding quarterly folds, 3 companies each, 24 predictions per model. Training target periods from 2021Q1 through t-2.',
  availability='Main uses financial and macro lags >=2 quarters, and train labels <=t-2. This assumed publication lag is NOT verified actual availability; no historical version/publication-date database. Retrospective only.',
  features=['ROA t-2,t-3,t-4','assets[t-2]/assets[t-3]-1','company ID (one-hot)','target calendar quarter (one-hot)'],
  macro_features=['USD/KZT quarter-end change q/q at t-2 (NBK)','quarter-average Brent change q/q at t-2 (EIA via FRED)'],
  excluded_features='No revenue/equity/GDP/CPI in this minimal comparison. Revenue excluded because of the KEGOC 2021 continuing/discontinued scope break. Not the exact annual feature matrix.',
  models={NAIVE:'ROA same quarter previous year (t-4)',GB:'sklearn GB absolute_error, 150 trees, lr=.03, depth=2, min leaf=5',CB:'CatBoost MAE, 200 iterations, lr=.03, depth=2, l2=5, plus the two macro features',LAST:'Diagnostic baseline: last assumed available ROA (t-2)',CB0:'Ablation control: exactly the same CatBoost without macro'},
  tuning='No tuning/early stopping or selection on the test; fixed compact settings. Categories encoded inside train fold. Numeric imputer fit only on train.',
  sensitivities={'reported_quarter_lag2':'Use directly reported 3-month profit where available; Q4 remains FY minus 9M. Annual sums can differ because of statement revisions.',
    'conditional_lag1':'Conditional sensitivity assumes prior quarter financial statements and labels already available, lags1,2,4. Not a genuine forecast at the start of the quarter.'},
  uncertainties='Only 8 test quarters and 3 firms. Paired cluster bootstrap resamples whole test quarters; descriptive, not proof of significance. 2025 annual outcomes previously known; exploratory pilot.',
  annual_comparison='Raw MAE is not comparable to annual MAE: target interval, firms, features, test years differ. Compare skill relative to each frequency baseline, not raw size.',
  quarter_adjustments='YTD differencing forces annual reconciliation arithmetically. Kcell 2020Q3 and 2024Q2 disagree with the directly reported quarter; both targets are tested. Q4 can contain audit/revision adjustments. KZTO FY2024 uses original current-year statement, unlike revised comparative in annual panel.',
  seed=SEED,versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'catboost':catboost.__version__},
  annual_hashes_before=frozen,input_sha256=sha(BASE/'reviewed_interim.csv'),code_sha256=sha(Path(__file__)))
 js('protocol.json',protocol)
 main_d,nums=features(raw,macro)
 altered=raw.copy();altered.loc[altered.year.eq(2024)&altered.quarter.eq(1),'roa']*=9
 pert,_=features(altered,macro)
 cols=nums+['fx_change_qoq','brent_change_qoq','company','quarter_cat']
 check('own_target_does_not_enter_features',main_d[main_d.period.eq('2024Q1')][cols].equals(pert[pert.period.eq('2024Q1')][cols]))
 preds=[];folds=[]
 for scenario,delay,reported in [('main_lag2',2,False),('reported_quarter_lag2',2,True),('conditional_lag1',1,False)]:
  p,f,d,n=run_scenario(raw,macro,scenario,delay,reported);preds.append(p);folds.append(f)
  save('features_'+scenario+'.csv',d)
 p=pd.concat(preds,ignore_index=True);f=pd.concat(folds,ignore_index=True)
 met=metrics(p,['scenario','model']);m=met[met.model.eq(NAIVE)][['scenario','MAE_pp']].rename(columns={'MAE_pp':'seasonal_MAE'})
 met=met.merge(m,on='scenario');met['skill_vs_seasonal_pct']=(1-met.MAE_pp/met.seasonal_MAE)*100
 bycompany=metrics(p,['scenario','company_name','model']);byyear=metrics(p,['scenario','year','model']);byq=metrics(p,['scenario','period','model'])
 mainp=p[p.scenario.eq('main_lag2')];wide=mainp.pivot(index=['period','company'],columns='model',values='ae_pp')
 rng=np.random.default_rng(SEED);boot=[]
 for model_name,reference in [(GB,NAIVE),(CB,NAIVE),(CB,CB0)]:
  dif=(wide[model_name]-wide[reference]).groupby(level='period').mean().to_numpy()
  bs=rng.choice(dif,(10000,len(dif)),replace=True).mean(axis=1)
  boot.append(dict(model=model_name,reference=reference,MAE_difference_pp=float(dif.mean()),lower=float(np.quantile(bs,.025)),upper=float(np.quantile(bs,.975)),test_quarters=len(dif)))
 check('same_24_test_cases_all_models',p.groupby(['scenario','model']).size().eq(24).all())
 check('all_forecasts_finite',np.isfinite(p[['actual','prediction']]).all().all())
 check('original_annual_files_unchanged',frozen=={p.name:sha(p) for p in ANNUAL.iterdir() if p.is_file()})
 for name,df in [('predictions.csv',p),('metrics.csv',met),('by_company.csv',bycompany),('by_year.csv',byyear),('by_quarter.csv',byq),('folds.csv',f),('checks.csv',pd.DataFrame(checks))]:save(name,df)
 js('paired_bootstrap.json',boot)
 js('results.json',dict(company=recs(raw),macro=recs(macro),metrics=recs(met),by_company=recs(bycompany),by_year=recs(byyear),by_quarter=recs(byq),predictions=recs(p),annual_reconciliation=recs(annual),bootstrap=boot,protocol=protocol,sources=sources))
 js('run_complete.json',dict(status='complete',rows=len(raw),main_test_cases=24,checks=len(checks),all_checks_passed=True,annual_unchanged=True))
 print(met.to_string(index=False));print('BY COMPANY\n',bycompany[bycompany.scenario.eq('main_lag2')].to_string(index=False));print('BOOT',boot)
if __name__=='__main__':main()
