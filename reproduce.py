"""Run from the repository root. No network or PDF access needed.

python reproduce.py                 # re-aggregate all saved predictions
python reproduce.py --fit            # refit recorded annual choices and quarterly models
python reproduce.py --nested         # repeat original annual nested selection (4 scenarios)
"""
from pathlib import Path
import os, sys, argparse, json
os.environ.setdefault('OMP_NUM_THREADS','2')
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'code'/'original'))
import numpy as np
import pandas as pd
from harmonize_dataset_20261016 import make_features
from retest_harmonized_models_20261016 import run, estimator, metrics
from run_annual_experiments_20261019 import specs, make_model

def same(a,b,keys):
    z=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    assert len(z)==len(a)==len(b)
    assert np.allclose(z.actual_a,z.actual_b,atol=1e-12)
    delta=float(np.max(np.abs(z.prediction_a-z.prediction_b)))
    assert delta<1e-8,delta
    return delta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',action='store_true');ap.add_argument('--nested',action='store_true');args=ap.parse_args()
    dest=BASE/'recomputed';dest.mkdir(exist_ok=True)
    checks=[]
    for folder in ['annual','annual_extensions','quarterly']:
        p=pd.read_csv(BASE/'results'/folder/'predictions.csv')
        if folder=='annual':p=p[p.year.between(2018,2024)]
        met=metrics(p,by=['model'] if folder=='annual_extensions' else ['scenario','model'])
        met.to_csv(dest/(folder+'_metrics.csv'),index=False)
        old=pd.read_csv(BASE/'results'/folder/('model_metrics.csv' if folder=='annual' else 'metrics.csv'))
        keys=['model'] if folder=='annual_extensions' else ['scenario','model']
        z=met.merge(old,on=keys,suffixes=('_a','_b'),validate='one_to_one')
        assert len(z)==len(old) and np.allclose(z.MAE_pp_a,z.MAE_pp_b,atol=1e-9)
        checks.append({'comparison':folder+' metrics','passed':True})
    c=pd.read_csv(BASE/'data/annual/company_financials.csv');m=pd.read_csv(BASE/'data/annual/macro_by_year.csv')
    f=make_features(c,m)
    if args.nested:
        scenarios=[('main_ifrs',f,1),('broad_scope_sensitivity',make_features(c,m,allow_scope_uncertain=True),1),('no_parent_overlap',make_features(c[~c.company.isin(['НК КазМунайГаз','Казахтелеком'])],m),1),('lag2_delay_sensitivity',make_features(c,m,lag=2),2)]
        expected=pd.read_csv(BASE/'results/annual/predictions.csv')
        for name,data,lag in scenarios:
            p,fold,sel,cv=run(data,name,lag,range(2018,2025))
            old=expected[expected.scenario.eq(name)&expected.year.between(2018,2024)]
            p.to_csv(dest/(name+'_nested_predictions.csv'),index=False)
            sel.to_csv(dest/(name+'_nested_choices.csv'),index=False)
            cv.to_csv(dest/(name+'_nested_inner_cv.csv'),index=False)
            delta=same(p,old,['scenario','model','company','year'])
            checks.append({'comparison':name+' nested','max_delta':delta})
    if args.fit:
        expected=pd.read_csv(BASE/'results/annual_extensions/predictions.csv')
        selected=pd.read_csv(BASE/'results/annual_extensions/selected_models.csv')
        sp={s['name']:s for s in specs()};rows=[]
        for r in selected.itertuples():
            if r.model not in sp:continue
            s=sp[r.model];parameter=float(r.parameter) if s['family']=='Ridge' else int(float(r.parameter))
            tr=f[f.year<r.year];te=f[f.year.eq(r.year)]
            # Preserve the original floating-point arithmetic order: subtract
            # percentage-point targets, not ratios before multiplication.
            target=100*tr.actual.to_numpy()
            if s['target']=='delta':target=target-100*tr.roa_lag.to_numpy()
            est=make_model(s,parameter);est.fit(tr,target)
            pred_pp=est.predict(te)
            if s['target']=='delta':pred_pp=pred_pp+100*te.roa_lag.to_numpy()
            pred=pred_pp/100
            rows.extend(dict(company=co,year=int(r.year),model=r.model,actual=float(a),prediction=float(h)) for co,a,h in zip(te.company,te.actual,pred))
        actual=pd.DataFrame(rows);old=expected[expected.model.isin(sp)]
        actual.to_csv(dest/'annual_extensions_refitted.csv',index=False)
        checks.append({'comparison':'annual original and all exploratory family fits','max_delta':same(actual,old,['model','company','year'])})
        import run_quarterly_pilot_20261019 as q
        raw=pd.read_csv(BASE/'data/quarterly/company_quarterly.csv');mac=pd.read_csv(BASE/'data/quarterly/macro_quarterly.csv')
        old=pd.read_csv(BASE/'results/quarterly/predictions.csv')
        for name,delay,reported in [('main_lag2',2,False),('reported_quarter_lag2',2,True),('conditional_lag1',1,False)]:
            pred,_,_,_=q.run_scenario(raw,mac,name,delay,reported)
            checks.append({'comparison':name+' fits','max_delta':same(pred,old[old.scenario.eq(name)],['scenario','model','company','period'])})
            pred.to_csv(dest/(name+'_refitted.csv'),index=False)
    (dest/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
