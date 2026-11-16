"""Recompute the article's descriptive annual macro ablation bands.

Uses frozen predictions, not refitted models. Resamples seven whole-year clusters;
does not preserve adjacent-year dependence and is not a formal significance test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
original=pd.read_csv(BASE/'results/annual/predictions.csv')
original=original[original.scenario.eq('main_ifrs') & original.year.between(2018,2024)]
extended=pd.read_csv(BASE/'results/annual_extensions/predictions.csv')
rng=np.random.default_rng(20261021)
rows=[]
for experiment,prefix,data,families in [
    ('original','',original,['Ridge','RandomForest','GradientBoosting','CatBoost']),
    ('exploratory','Loss | ',extended,['GradientBoosting absolute_error','CatBoost MAE'])]:
    for family in families:
        left=data[data.model.eq(prefix+family+' internal')]
        right=data[data.model.eq(prefix+family+' + macro')]
        merged=left.merge(right,on=['company','year'],suffixes=('_internal','_macro'),validate='one_to_one')
        assert len(merged)==88
        merged['difference']=100*(abs(merged.prediction_macro-merged.actual_macro)-abs(merged.prediction_internal-merged.actual_internal))
        groups=[g.difference.to_numpy() for _,g in merged.groupby('year',sort=True)]
        simulations=np.array([np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(10000)])
        lower,upper=np.quantile(simulations,[.025,.975])
        rows.append(dict(experiment=experiment,family=family,N=len(merged),delta_MAE_pp=merged.difference.mean(),lower_pp=lower,upper_pp=upper,years_macro_better=sum(g.mean()<0 for g in groups)))
actual=pd.DataFrame(rows)
expected=pd.read_csv(BASE/'results/comparisons/paired_macro_bootstrap.csv')
cols=['delta_MAE_pp','lower_pp','upper_pp']
assert actual[['experiment','family']].equals(expected[['experiment','family']])
assert np.allclose(actual[cols],expected[cols],rtol=0,atol=1e-10)
dest=BASE/'recomputed';dest.mkdir(exist_ok=True)
actual.to_csv(dest/'paired_macro_bootstrap.csv',index=False)
print('All six annual macro comparisons and their descriptive bands reproduced.')

quarterly=pd.read_csv(BASE/'results/quarterly/predictions.csv')
quarterly=quarterly[quarterly.scenario.eq('main_lag2')].copy()
quarterly['recomputed_error']=100*abs(quarterly.prediction-quarterly.actual)
wide=quarterly.pivot(index=['period','company'],columns='model',values='recomputed_error')
rng=np.random.default_rng(20261019)
rows=[]
for model,reference in [('GB MAE internal','Seasonal naive'),('CatBoost MAE + macro','Seasonal naive'),('CatBoost MAE + macro','CatBoost MAE internal (control)')]:
    difference=(wide[model]-wide[reference]).groupby(level='period').mean().to_numpy()
    sims=rng.choice(difference,(10000,len(difference)),replace=True).mean(axis=1)
    rows.append(dict(model=model,reference=reference,MAE_difference_pp=float(difference.mean()),lower=float(np.quantile(sims,.025)),upper=float(np.quantile(sims,.975)),test_quarters=len(difference)))
actual=pd.DataFrame(rows)
expected=pd.DataFrame(json.loads((BASE/'results/quarterly/paired_bootstrap.json').read_text('utf-8')))
assert actual[['model','reference']].equals(expected[['model','reference']])
cols=['MAE_difference_pp','lower','upper']
assert np.allclose(actual[cols],expected[cols],rtol=0,atol=1e-10)
actual.to_csv(dest/'quarterly_paired_bootstrap.csv',index=False)
print('All three quarterly comparisons and their descriptive bands reproduced.')
