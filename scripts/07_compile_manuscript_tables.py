"""07 - Build authoritative manuscript tables and a compact key-number sheet.

All turbulent-flux statistics use one common valid mask per glacier requiring
Qh and Qe observations plus both BT and calibrated-katabatic components. This
ensures Qh and Qe use exactly the same hours (Peyto: after final EC QC).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from project_config import PROJECT_ROOT, PEYTO_RAW, ATHA_RAW, INTERIM, TABLES, FIGURES
# Import shared modules from the shipped scripts/ directory (not from a machine-specific root).
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline_common import INTERIM,TABLES,require
TABLES.mkdir(parents=True,exist_ok=True)


def metric(obs,mod,mask):
    o=pd.to_numeric(obs,errors='coerce').to_numpy(float)[mask]; m=pd.to_numeric(mod,errors='coerce').to_numpy(float)[mask]
    return {'n':len(o),'mean_obs':float(np.mean(o)),'mean_mod':float(np.mean(m)),'sd_mod':float(np.std(m,ddof=1)),
            'bias':float(np.mean(m-o)),'r':float(np.corrcoef(o,m)[0,1]),'rmse':float(np.sqrt(np.mean((m-o)**2)))}


def flux_table():
    rows=[]
    specs={
      'Athabasca':('Athabasca_melt_scenarios_hourly.csv','Qh_EC','Qe_EC'),
      'Peyto':('Peyto_melt_scenarios_hourly.csv','EC2QH','AdjEC2QE')}
    for glacier,(fn,qho,qeo) in specs.items():
        d=pd.read_csv(require(INTERIM/fn))
        cols=[qho,qeo,'Qh_Bulk','Qe_Bulk','Qh_Kata_T6_preserveEa_Kbcal','Qe_Kata_T6_preserveEa_Kbcal']
        arr=[pd.to_numeric(d[c],errors='coerce').to_numpy(float) for c in cols]
        mask=np.logical_and.reduce([np.isfinite(x) for x in arr])
        print(f'{glacier}: {mask.sum()} common flux-comparison hours')
        for flux,obs,bulk,kata in [('Qh',qho,'Qh_Bulk','Qh_Kata_T6_preserveEa_Kbcal'),('Qe',qeo,'Qe_Bulk','Qe_Kata_T6_preserveEa_Kbcal')]:
            obsvals=pd.to_numeric(d[obs],errors='coerce').to_numpy(float)[mask]
            rows.append({'site':glacier,'flux':flux,'method':'EC','n':len(obsvals),'mean':float(np.mean(obsvals)),'sd':float(np.std(obsvals,ddof=1)),'bias':np.nan,'r':np.nan,'rmse':np.nan})
            for method,col in [('BT',bulk),('Kata',kata)]:
                mm=metric(d[obs],d[col],mask); rows.append({'site':glacier,'flux':flux,'method':method,'n':mm['n'],'mean':mm['mean_mod'],'sd':mm['sd_mod'],'bias':mm['bias'],'r':mm['r'],'rmse':mm['rmse']})
        # total flux, useful for Results text
        obsT=d[qho]+d[qeo]; btT=d.Qh_Bulk+d.Qe_Bulk; kaT=d.Qh_Kata_T6_preserveEa_Kbcal+d.Qe_Kata_T6_preserveEa_Kbcal
        for method,series in [('EC',obsT),('BT',btT),('Kata',kaT)]:
            if method=='EC':
                vals=pd.to_numeric(series,errors='coerce').to_numpy(float)[mask]; rows.append({'site':glacier,'flux':'Qh+Qe','method':'EC','n':len(vals),'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)),'bias':np.nan,'r':np.nan,'rmse':np.nan})
            else:
                mm=metric(obsT,series,mask); rows.append({'site':glacier,'flux':'Qh+Qe','method':method,'n':mm['n'],'mean':mm['mean_mod'],'sd':mm['sd_mod'],'bias':mm['bias'],'r':mm['r'],'rmse':mm['rmse']})
    out=pd.DataFrame(rows); out.to_csv(TABLES/'Table_flux_primary.csv',index=False); return out


def main():
    flux=flux_table()
    melt=pd.read_csv(require(TABLES/'Measured_melt_comparison_summary.csv'))
    melt=melt[melt.scenario.isin(['EC','Bulk','Kata_T6_preserveEa_Kbcal'])].copy(); melt['method']=melt.scenario.map({'EC':'EC','Bulk':'BT','Kata_T6_preserveEa_Kbcal':'Kata'})
    melt.to_csv(TABLES/'Table_melt_primary.csv',index=False)
    kb=pd.read_csv(require(TABLES/'Kb_forcing_comparison.csv'))
    kb=kb[kb.scenario.isin(['Kb_0.01','Kb_mean_match'])].copy(); kb.to_csv(TABLES/'Table_kb_sensitivity.csv',index=False)

    key=[]
    for glacier in ['Athabasca','Peyto']:
        forcing='revised_T6m_preserve_ea1.5m' if glacier=='Athabasca' else 'revised_T6m_preserve_ea2m'
        hit=kb[(kb.glacier==glacier)&(kb.forcing==forcing)&(kb.scenario=='Kb_mean_match')].iloc[0]
        key.append({'glacier':glacier,'metric':'calibrated_Kb_m_s','value':hit.Kb_m_s})
        for method in ['BT','Kata']:
            r=flux[(flux.site==glacier)&(flux.flux=='Qh+Qe')&(flux.method==method)].iloc[0]
            key += [{'glacier':glacier,'metric':f'{method}_total_RMSE_W_m2','value':r.rmse},{'glacier':glacier,'metric':f'{method}_total_bias_W_m2','value':r.bias}]
        for method in ['EC','BT','Kata']:
            r=melt[(melt.glacier==glacier)&(melt.method==method)].iloc[0]
            key.append({'glacier':glacier,'metric':f'{method}_melt_percent_error','value':r.percent_error})
    pd.DataFrame(key).to_csv(TABLES/'Manuscript_key_numbers.csv',index=False)
    print('\nTable_flux_primary.csv\n',flux.to_string(index=False)); print('\nManuscript_key_numbers.csv\n',pd.DataFrame(key).to_string(index=False))

if __name__=='__main__': main()
