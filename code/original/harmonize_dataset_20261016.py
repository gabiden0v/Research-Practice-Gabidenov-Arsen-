"""Versioned harmonisation. Original files remain immutable; no fabricated values."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'outputs/dataset_audit_2026-10-16_continued'
OUT=ROOT/'outputs/dataset_harmonized_2026-10-16'
KEY=['Компания','Год']
RAW=['Выручка','Чистая прибыль','Активы','Капитал']
AIR_SALES={2015:726800+19586,2016:589926+15685,2017:718178+19666,
           2018:810353+20703,2019:824952+21145,2020:358413+24561,
           2021:715794+33570,2022:998120+22124,2023:1143596+22519,2024:1246044+26303}
SECTOR={
 'Air Astana':'Транспорт','Қазақстан темір жолы':'Транспорт',
 'KEGOC':'Электроэнергетика','Самрук-Энерго':'Электроэнергетика',
 'АК Алтыналмас':'Горнодобыча и металлургия','Казатомпром':'Горнодобыча и металлургия','ТНК Казхром':'Горнодобыча и металлургия',
 'Кселл':'Телекоммуникации','Казахтелеком':'Телекоммуникации',
 'НК КазМунайГаз':'Нефть и газ','КазТрансОйл':'Нефть и газ','QazaqGaz':'Нефть и газ',
 'Озенмунайгаз':'Нефть и газ','Мангистаумунайгаз':'Нефть и газ','Тенгизшевройл':'Нефть и газ'}
# Broad relationship clusters for sensitivity, not legal historical control assertions.
GROUP={c:('KMG-related' if c in ['НК КазМунайГаз','КазТрансОйл','QazaqGaz','Озенмунайгаз','Мангистаумунайгаз','Тенгизшевройл']
          else 'Telecom-related' if c in ['Казахтелеком','Кселл'] else c) for c in SECTOR}
# Based on source-review evidence, fixed BEFORE inspecting model scores.
REV_BREAKS={'НК КазМунайГаз':{2014,2018,2021,2023},'KEGOC':{2020},
            'Казахтелеком':{2018,2019,2024},'АК Алтыналмас':{2019,2020},
            'Казатомпром':{2017,2018},'Озенмунайгаз':{2023}}
ASSET_BREAKS={'Air Astana':{2016,2019},'Кселл':{2017,2019},'ТНК Казхром':{2018},
              'НК КазМунайГаз':{2021,2023},'Казахтелеком':{2018,2019,2024,2025},
              'Самрук-Энерго':{2023},'Озенмунайгаз':{2023}}
KNOWN_DISCONT={'НК КазМунайГаз':set(range(2014,2018)), 'KEGOC':{2020,2021},
              'ТНК Казхром':{2017,2018},'Казатомпром':{2018},
              'Казахтелеком':{2012,2024,2025},'Мангистаумунайгаз':{2009}}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def json_write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write_csv(name,df): df.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def make_features(company,macro,lag=1,allow_scope_uncertain=False):
    """Exact year-key lags. No dependence on future rows or target-year flags."""
    d=company.sort_values(['company','year']).copy().reset_index(drop=True)
    ix=d.set_index(['company','year'])
    mi=macro.set_index('year')
    rows=[]
    for r in d.to_dict('records'):
        c,y=r['company'],r['year']; py=y-lag
        eligible=r['broad_eligible'] if allow_scope_uncertain else r['main_eligible']
        if not eligible or (c,py) not in ix.index: continue
        p=ix.loc[(c,py)]
        p_ok=p['broad_eligible'] if allow_scope_uncertain else p['main_eligible']
        if not p_ok: continue
        q=ix.loc[(c,py-1)] if (c,py-1) in ix.index else None
        q_ok=q is not None and (q['broad_eligible'] if allow_scope_uncertain else q['main_eligible'])
        rev_growth=asset_growth=np.nan
        if q_ok:
            if py not in REV_BREAKS.get(c,set()) and np.isfinite(p['sales_revenue']) and np.isfinite(q['sales_revenue']) and q['sales_revenue']>0:
                rev_growth=p['sales_revenue']/q['sales_revenue']-1
            if py not in ASSET_BREAKS.get(c,set()) and q['assets']>0:
                asset_growth=p['assets']/q['assets']-1
        hist=[]
        for hy in range(py-2,py+1):
            if (c,hy) in ix.index:
                h=ix.loc[(c,hy)]
                if h['broad_eligible'] if allow_scope_uncertain else h['main_eligible']:
                    hist.append(h['net_profit']/h['assets'])
        rec=dict(company=c,year=y,sector=r['sector_group'],group=r['relationship_cluster'],
            actual=r['net_profit']/r['assets'],roa_lag=p['net_profit']/p['assets'],
            equity_lag=p['equity']/p['assets'],asset_growth_lag=asset_growth,revenue_growth_lag=rev_growth,
            mean3=float(np.mean(hist)),feature_year=py,macro_year=py,
            # For filtering only: never a predictor. Used in explicitly retrospective sensitivity.
            target_break=(y in REV_BREAKS.get(c,set()) or y in ASSET_BREAKS.get(c,set())),
            target_discontinued=y in KNOWN_DISCONT.get(c,set()))
        for field in ['gdp_growth','inflation','fx_change','brent_change']:
            rec[field]=mi.loc[py,field] if py in mi.index else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(['company','year']).reset_index(drop=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=json.loads((ROOT/'outputs/dataset_full_review_2026-10-16/row_evidence.json').read_text(encoding='utf-8'))
    ev={(x['company'],x['year']):x for x in evidence}
    core=pd.read_csv(OLD/'company_financials_corrected.csv').fillna('')
    oldhash={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'company_financials_corrected.csv',OLD/'panel_corrected.csv',OLD/'Датасет_15_компаний_исправленный.xlsx']}
    changes=[]; registry=[]; records=[]
    ids={c:f'C{i:02d}' for i,c in enumerate(sorted(core['Компания'].unique()),1)}
    for _,r in core.iterrows():
        c,y=r['Компания'],int(r['Год']); e=ev[c,y]; sid=f'{ids[c]}_{y}'
        report_revenue=float(r['Выручка']); sales=report_revenue
        note=[]; adjustments=[]
        if c=='Air Astana' and y==2015:
            assert report_revenue==738100
            report_revenue=761747
            adjustments.append(dict(field='reported_revenue',old=738100,new=761747,type='confirmed_error',source='air2015.pdf PDF 62, printed 120'))
        if c=='Air Astana':
            sales=AIR_SALES[y]
            adjustments.append(dict(field='sales_revenue',old=report_revenue,new=sales,type='definition_harmonisation',
                source=e['source'],pages=e['pages'],rule='Passenger revenue + cargo and mail revenue. Other income and sale/leaseback gains excluded.'))
            note.append('Продажи: пассажиры, грузы и почта. Прочие доходы отделены.')
        if c=='ТНК Казхром' and y==2017:
            sales=702097306
            adjustments.append(dict(field='sales_revenue',old=report_revenue,new=sales,type='restated_continuing_revenue',
                source='https://kase.kz/files/emitters/KZCR/kzcrp_2018_rus.pdf',pages='PDF 39',
                rule='Continuing revenue comparative from 2018 report; original 2017 total retained separately.'))
        if c=='Тенгизшевройл':
            sales=np.nan
            note.append('Общий доход не подменяет продажи. Учет по партнерскому соглашению, вне основного МСФО-теста.')
        scope='Консолидированная'
        scope_uncertain=c=='Озенмунайгаз' and y<=2022
        if c=='Air Astana' and y<=2018:
            scope='Компания (до групповой отчетности)'
        if scope_uncertain:
            scope='Финансовая отчетность; периметр не уточнен'
            note.append('Периметр не подтвержден; только дополнительный сценарий.')
        partial=c=='Озенмунайгаз' and y==2012
        if partial: note.append('Неполный период: 25.02–31.12.2012, не годовое наблюдение для модели.')
        mmg_sep=c=='Мангистаумунайгаз' and y==2025
        if mmg_sep:
            scope='Отдельная'
            note.append('Не объединять с консолидированной историей в модели.')
        if y in REV_BREAKS.get(c,set()): note.append('Темп роста продаж через этот год не рассчитывается: разрыв базы.')
        if y in ASSET_BREAKS.get(c,set()): note.append('Темп роста активов через этот год не рассчитывается: разрыв базы.')
        if y in KNOWN_DISCONT.get(c,set()): note.append('Общая прибыль включает прекращенную деятельность.')
        if c=='АК Алтыналмас' and y==2020: note.append('Прибыль из версии отчета 2021; расхождение с обзором 2020 на 90 тыс. KZT не разрешено.')
        if c=='КазТрансОйл' and y in [2023,2024]: note.append('Разные версии сравнительных остатков капитала; не гарантия доступности в прошлом.')
        main_ok=not(c=='Тенгизшевройл' or partial or mmg_sep or scope_uncertain)
        broad_ok=not(c=='Тенгизшевройл' or partial or mmg_sep)
        reasons=[]
        if c=='Тенгизшевройл':reasons.append('special_accounting_basis')
        if partial:reasons.append('partial_year')
        if mmg_sep:reasons.append('standalone_vs_group')
        if scope_uncertain:reasons.append('unconfirmed_scope')
        records.append(dict(company_id=ids[c],company=c,year=y,sector=r['Сектор'],sector_group=SECTOR[c],
            relationship_cluster=GROUP[c],currency=r['Валюта'],unit='тыс.',sales_revenue=sales,
            reported_revenue=report_revenue,net_profit=float(r['Чистая прибыль']),assets=float(r['Активы']),equity=float(r['Капитал']),
            reporting_scope=scope,accounting_basis='Partnership agreement' if c=='Тенгизшевройл' else 'IFRS',
            period_start=f'{y}-02-25' if partial else f'{y}-01-01',period_end=f'{y}-12-31',
            main_eligible=main_ok,broad_eligible=broad_ok,exclusion_reason='; '.join(reasons),
            source_id=sid,notes=' '.join(note),source_title=Path(e['source']).name if not e['source'].startswith('http') else e['source']))
        source=dict(e);source.update(source_id=sid,company_id=ids[c],adjustments=adjustments,
            publication_date=None,vintage_policy='Frozen source-selected retrospective history; no point-in-time guarantee.',
            harmonized_revenue_definition=('Passenger, cargo and mail sales; other income excluded' if c=='Air Astana' else
                'Not harmonised, missing sales value; reported total income retained' if c=='Тенгизшевройл' else 'Source sales / continuing revenue; disclosed structural breaks are masked in growth features'),
            profit_definition='Total net profit after tax including non-controlling interests and discontinued operations; not OCI',
            equity_definition='Total equity including non-controlling interests',
            limitations=note)
        registry.append(source)
        for a in adjustments:changes.append(dict(company=c,year=y,**a))
    d=pd.DataFrame(records).sort_values(['company','year']).reset_index(drop=True)
    assert len(d)==196 and not d.duplicated(['company','year']).any()
    assert int(d.main_eligible.sum())==165
    assert d[['reported_revenue','net_profit','assets','equity']].notna().all().all()
    assert d[d.company.ne('Тенгизшевройл')].sales_revenue.notna().all()
    write_csv('company_financials.csv',d)
    # Human view: one company-year per row, no macro or model features.
    human=d[['company','year','sector','currency','sales_revenue','net_profit','assets','equity','reported_revenue','reporting_scope','accounting_basis','main_eligible','source_id','notes','source_title']].copy()
    human.columns=['Компания','Год','Сектор','Валюта','Выручка продаж, тыс.','Чистая прибыль всего, тыс.','Активы, тыс.','Капитал всего, тыс.','Доходы по исходной строке, тыс.','Периметр','Основа учета','В основной выборке','ID источника','Ограничения','Источник']
    human['В основной выборке']=human['В основной выборке'].map({True:'Да',False:'Нет'})
    write_csv('company_view.csv',human)
    json_write('source_registry.json',registry);json_write('changes.json',changes)
    # Store external levels independently. Never silently substitute end-year FX.
    series={};meta={}
    for name,file in [('gdp_growth','wb_gdp.json'),('inflation','wb_inflation.json'),('fx_average','wb_fx.json')]:
        ob=json.loads((ROOT/'tmp'/file).read_text(encoding='utf-8'))
        series[name]={int(x['date']):x['value'] for x in ob[1] if x.get('value') is not None}
        meta[name]=dict(indicator=ob[1][0]['indicator']['id'],lastupdated=ob[0]['lastupdated'],cache_sha256=sha(ROOT/'tmp'/file))
    br=pd.read_csv(ROOT/'tmp/brent.csv'); series['brent_average']=dict(zip(pd.to_datetime(br.observation_date).dt.year,br.ACOILBRENTEU))
    macro=pd.DataFrame({'year':range(2004,2026),'country':'KAZ'})
    for col,mapping in series.items():macro[col]=macro.year.map(mapping)
    for col in ['gdp_growth','inflation']:macro[col]/=100
    for col,new in [('fx_average','fx_change'),('brent_average','brent_change')]:
        macro[new]=macro.year.map(series[col])/((macro.year-1).map(series[col]))-1
    macro['wb_vintage']='2026-07-13'
    macro['source_id']='WB_KAZ_20260713_FRED_BRENT_cache'
    write_csv('macro_by_year.csv',macro)
    mh=macro[['year','country','gdp_growth','inflation','fx_average','brent_average','wb_vintage','source_id']].copy()
    mh.columns=['Год','Страна','Рост реального ВВП','Среднегодовая инфляция','Средний курс KZT/USD','Brent, USD/баррель','Версия World Bank','ID источника']
    mh['Источник']='World Bank: NY.GDP.MKTP.KD.ZG, FP.CPI.TOTL.ZG, PA.NUS.FCRF; FRED: ACOILBRENTEU'
    write_csv('macro_view.csv',mh)
    json_write('macro_sources.json',{'world_bank':meta,'brent':{'url':'https://fred.stlouisfed.org/series/ACOILBRENTEU','cache_sha256':sha(ROOT/'tmp/brent.csv')},
        'urls':['https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=KZ','https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=KZ','https://data.worldbank.org/indicator/PA.NUS.FCRF?locations=KZ'],
        'limitations':'Revised retrospective vintage, not historical release-time data. Missing source values stay missing.'})
    f=make_features(d,macro)
    write_csv('model_features.csv',f)
    json_write('data_dictionary.json',{
      'grain':'company-year; financial fields in thousands of the specified original currency',
      'sales_revenue':'Air Astana: passenger + cargo/mail. Kazchrome 2017: restated continuing sales. TCO: unavailable, not zero. Other companies: source sales/continuing revenue.',
      'reported_revenue':'Corrected amount from the source line previously called revenue; may include other income. Kept for audit, not substituted for sales.',
      'net_profit':'Total after-tax profit including NCI and discontinued operations; not owners-only or OCI.',
      'assets':'Total assets at year end.','equity':'Total equity including NCI at year end.',
      'actual':'Target ROA = total net profit / year-end assets, ratio units; not average-assets or continuing-operations ROA.',
      'features':['lagged ROA','lagged total equity/assets','lagged asset growth','lagged sales growth'],
      'margin':'Deliberately omitted: total profit and continuing revenue can have different scopes.',
      'publication_date':'Unknown unless separately evidenced. No fabricated as-of dates.',
      'main_eligible':'IFRS; full-year; comparable known reporting scope. Does not mean all structural changes have been undone.',
      'structural_breaks':'Growth through identified source/perimeter breaks is missing, not zero. Lagged values only; target-year event flags never enter predictors.',
      'currency':'No automatic KZT/USD conversion; models use dimensionless ratios and changes within a company. Nominal totals cannot be summed across currencies.',
      'relationship_cluster':'Conservative related-company grouping for sensitivity, not a time-invariant assertion of legal ownership.',
      'restatement_policy':'Frozen retrospective selected sources with documented exceptions. Full historical release reconstruction remains unresolved.'})
    summary=dict(rows=len(d),companies=d.company.nunique(),main_rows=int(d.main_eligible.sum()),
        main_companies=d[d.main_eligible].company.nunique(),broad_rows=int(d.broad_eligible.sum()),
        missing_sales=int(d.sales_revenue.isna().sum()),model_rows=len(f),
        test_rows_2018_2024=int(f.year.between(2018,2024).sum()),
        numeric_error_fixes=sum(a['type']=='confirmed_error' for a in changes),
        definition_changes=sum(a['type']!='confirmed_error' for a in changes),
        old_sha256=oldhash,exclusions=d[~d.main_eligible].groupby('exclusion_reason').size().to_dict())
    json_write('preparation_summary.json',summary)
    assert all(sha(ROOT/k)==v for k,v in oldhash.items())
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
