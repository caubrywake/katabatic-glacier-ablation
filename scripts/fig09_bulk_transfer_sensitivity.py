# %% Figure 9 v4 — B10: final Code 12 constants and retained validation periods
"""Recalculate Figure 9 roughness/measurement-height BULK-TRANSFER sensitivity.

Changes relative to legacy Code 06
----------------------------------
- Mapped to final Code 12: Lv = 2.5 MJ/kg, Lf = 334 kJ/kg, Ls = Lv + Lf.
- Positive vapour mass flux = liquid condensation, not negative solid ablation.
- Sublimation of ice is added to melt to obtain surface ABLATION.
- Cumulative sensitivity is integrated only over final SR50 intervals:
    Peyto four intervals: (12,61], (274,372], (491,582], (732,765]
    Athabasca: [0,288) (last grid sample is SR50 endpoint only).
- Retain the reference settings: z0m = 2.41 mm, height 2 m at Peyto,
  ice-profile 1.5 m at Athabasca. Athabasca reference height sensitivity
  uses profile forcing, unlike the primary AWS-driven bulk run.
- Keep sensible heat RMSE based on paired valid EC turbulent-flux observations.
- Produce source and quality-control tables and a reproducible 4-panel figure.

Scientific limitations:
- 4/6 m Peyto RH is a separate exploratory sensitivity, not validated forcing.
  These profiles may have sensor issues. Only 0 <= RH <= 100 is admitted.
- Athabasca 6 m profile is omitted owing to unreliable RH; 1.5 m wind
  missing/zero values are filled using regression against AWS wind as in Code 06.
- The Athabasca AWS bulk series is missing its first hour; no zero-filling.
- Experiments compare parameter variants on identical valid hours. Diagnostic
  source coverage is reported in the audit table.
- No source data or existing Code 06 tables/figures are overwritten.

Run as .py or entire VS Code cell. Requires pandas, numpy, matplotlib and
src/turbulent_fluxes.py from the existing katabatic project.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_config import PROJECT_ROOT as DEFAULT_ROOT
Z0_VALUES = (0.0005, 0.001, 0.00241, 0.005, 0.01)
PEYTO_HEIGHTS = (1.0, 2.0, 4.0, 6.0)
ATHA_HEIGHTS = (0.75, 1.5, 3.0, 4.5)
LF = 334_000.0
LV = 2_500_000.0
LS = LF + LV
DT = 3600.0
SEGMENTS = ((12, 61), (274, 372), (491, 582), (732, 765))
STEM = 'Fig09_bulk_ablation_sensitivity_v4'
COLORS = {'Peyto':'#2367aa', 'Athabasca':'#e56b1f'}
FONT_PT = 14


def find_root(explicit_root=None):
    root = Path(explicit_root).expanduser().resolve() if explicit_root is not None else DEFAULT_ROOT
    if not (root / "data" / "interim" / "Peyto_ablation_partition_hourly.csv").exists():
        raise FileNotFoundError(f"Missing stage 08 output under {root}; run stage 08 first")
    return root


def source_paths(root):
    return (root / 'data' / 'interim', root / 'data' / 'raw' / 'athabasca',
            root / 'figures', root / 'tables')


def load_csv(path):
    if not path.is_file(): raise FileNotFoundError(f'Missing input: {path}')
    return pd.read_csv(path,parse_dates=['time']).set_index('time').sort_index()


def retained_mask(glacier,n):
    m=np.zeros(n,dtype=bool)
    if glacier=='Peyto':
        if n!=768: raise ValueError('Expected 768 Peyto rows; check source and indexing.')
        for i,j in SEGMENTS: m[i+1:j+1]=True
    else:
        if n!=289: raise ValueError('Expected 289 Athabasca rows; check source and indexing.')
        m[:288]=True
    return m


def ablation_rates(qn,qp,qh,qe):
    """Code 12 v3 positive_phase='liquid', assuming flux energy used Lv."""
    qn=np.asarray(qn,float);qp=np.asarray(qp,float)
    qh=np.asarray(qh,float);qe=np.asarray(qe,float)
    vapor_mass=qe/LV
    qe_phase=np.where(vapor_mass<0,vapor_mass*LS,vapor_mass*LV)
    qm=qn+qp+qh+qe_phase
    melt=np.maximum(qm,0)*DT/LF
    sublimation=np.maximum(-vapor_mass,0)*DT
    return melt+sublimation,melt,sublimation


def stats(obs,mod,other=None):
    obs=np.asarray(obs,float);mod=np.asarray(mod,float)
    valid=np.isfinite(obs)&np.isfinite(mod)
    if other is not None:valid &= np.asarray(other,bool)
    if valid.sum()<2:return {'n':int(valid.sum()),'Qh_RMSE_W_m2':np.nan}
    return {'n':int(valid.sum()),'Qh_RMSE_W_m2':float(np.sqrt(np.mean((mod[valid]-obs[valid])**2)))}


def read_atha_profile(path,index):
    d=pd.read_csv(path,sep='\t',skiprows=[0,2,3],na_values=['NAN','NaN','nan'])
    t=pd.to_datetime(d['TIMESTAMP'],errors='coerce')
    out=pd.DataFrame(index=pd.DatetimeIndex(t))
    for i,h in enumerate((0.75,1.5,3.0,4.5,6.0),1):
        tag=str(h).replace('.','p')
        for col,sensor in [('T','AirTemp_Avg'),('RH','RH_Avg'),('U','WindSpeed_Avg')]:
            out[f'{col}_{tag}']=pd.to_numeric(d[f'{sensor}({i})'],errors='coerce').to_numpy(float)
    return out[~out.index.isna()].resample('1h').mean().reindex(index)


def fill_atha_1p5_wind(profile,aws):
    up=pd.to_numeric(profile['U_1p5'],errors='coerce')
    ua=pd.to_numeric(aws['U_AWS_ms'],errors='coerce')
    valid=np.isfinite(up)&(up>0)&np.isfinite(ua)
    if valid.sum()<3:raise ValueError('Insufficient good 1.5 m wind for regression.')
    slope,intercept=np.polyfit(ua[valid],up[valid],1)
    filled=up.copy()
    bad=(~np.isfinite(up))|(up<=0)
    use=bad&np.isfinite(ua)
    filled.loc[use]=(intercept+slope*ua.loc[use]).clip(lower=0)
    print(f'Athabasca 1.5-m profile wind: regression filled {int(use.sum())} hourly values')
    return filled


def compute_experiment(df,glacier,kind,bulk_transfer,profile=None):
    n=len(df); period=retained_mask(glacier,n)
    out=[]; scenarios=[]
    if glacier=='Peyto':
        obs=df['EC2QH']; ec_qe=df['AdjEC2QE']; z0T=df['T_2m_C'];z0RH=df['RH_2m_pct'];z0U=df['u_2m'];z0z=2.0;alt=2169
    else:
        obs=df['Qh_EC'];ec_qe=df['Qe_EC'];z0T=df['T_AWS_C'];z0RH=df['RH_AWS_pct'];z0U=df['U_AWS_ms'];z0z=1.5;alt=2175
    ec_common=np.isfinite(obs.to_numpy(float))&np.isfinite(ec_qe.to_numpy(float))
    options=Z0_VALUES if kind=='roughness' else (PEYTO_HEIGHTS if glacier=='Peyto' else ATHA_HEIGHTS)
    for val in options:
        if kind=='roughness':
            T=z0T.to_numpy(float);RH=z0RH.to_numpy(float);U=z0U.to_numpy(float)
            h=z0z;z0=float(val)
        elif glacier=='Peyto':
            tag=f'{int(val)}m'
            T=df[f'T_{tag}_C'].to_numpy(float);RH=df[f'RH_{tag}_pct'].to_numpy(float)
            U=df[f'u_{tag}'].to_numpy(float);h=float(val);z0=0.00241
        else:
            tag=str(val).replace('.','p')
            T=profile[f'T_{tag}'].to_numpy(float)
            RH=profile[f'RH_{tag}'].to_numpy(float)
            U=(profile['U_1p5_filled'] if val==1.5 else profile[f'U_{tag}']).to_numpy(float)
            h=float(val);z0=0.00241
        RH=np.where((RH>=0)&(RH<=100),RH,np.nan)
        mdl=bulk_transfer(T+273.15,U,RH,altitude_m=alt,z_m=h,z0m=z0)
        qh=np.asarray(mdl['Qh'],float);qe=np.asarray(mdl['Qe'],float)
        ablation,melt,sub=ablation_rates(df['Qn'].to_numpy(float),df['Qp'].to_numpy(float),qh,qe)
        scenarios.append((float(val),qh,qe,ablation,melt,sub))

    # Critical: compare configurations over EXACTLY the same valid intervals.
    # In the old script height experiments had different coverage; that can
    # cause artificial apparent effects of changing sensor height.
    common_ablation_hours=period.copy()
    common_rmse_hours=ec_common.copy()
    for val,qh,qe,abl,melt,sub in scenarios:
        common_ablation_hours &= np.isfinite(abl)
        common_rmse_hours &= np.isfinite(qh)

    for val,qh,qe,ablation,melt,sub in scenarios:
        valid=common_ablation_hours
        sh=stats(obs.to_numpy(float),qh,common_rmse_hours)
        out.append({'glacier':glacier,'experiment':kind,
          'z0_m':float(val) if kind=='roughness' else 0.00241,
          'height_m':float(val) if kind=='height' else z0z,
          'value':float(val),'n_Qh_RMSE':sh['n'],'Qh_RMSE_W_m2':sh['Qh_RMSE_W_m2'],
          'Ablation_mmwe':float(np.sum(ablation[valid])) if valid.any() else np.nan,
          'Melt_mmwe':float(np.sum(melt[valid])) if valid.any() else np.nan,
          'Sublimation_mmwe':float(np.sum(sub[valid])) if valid.any() else np.nan,
          'n_valid_ablation_hours':int(valid.sum()),
          'n_possible_valid_ablation_hours':int(np.sum(period & np.isfinite(ablation))),
          'n_expected_ablation_hours':int(period.sum())})
    result=pd.DataFrame(out)
    baseline=0.00241 if kind=='roughness' else (2.0 if glacier=='Peyto' else 1.5)
    row=result.iloc[int(np.argmin(abs(result.value-baseline)))]
    if not np.isclose(row.value,baseline,atol=1e-9):raise AssertionError('No baseline in sensitivity grid')
    ref=float(row.Ablation_mmwe)
    result['Reference_ablation_mmwe']=ref
    result['Relative_ablation_change_pct']=(result.Ablation_mmwe/ref-1)*100
    return result


def validate_against_code12(df,glacier,pz,check=True):
    existing=ablation_rates(df['Qn'].to_numpy(float),df['Qp'].to_numpy(float),
              df['Qh_Bulk'].to_numpy(float),df['Qe_Bulk'].to_numpy(float))[0]
    code12=df['Ablation_Bulk_mm_h'].to_numpy(float)
    both=np.isfinite(existing)&np.isfinite(code12)
    assert both.any()
    maxdiff=float(np.max(np.abs(existing[both]-code12[both])))
    mask=retained_mask(glacier,len(df))
    code12_sum=float(np.nansum(code12[mask]));base=float(pz.loc[np.isclose(pz.value,0.00241),'Ablation_mmwe'].iloc[0])
    print(f'{glacier} baseline: recalculated={base:.5f}, Code12={code12_sum:.5f}, max hourly difference={maxdiff:.3g}')
    if check and (abs(base-code12_sum)>1.e-5 or maxdiff>1.e-8):
        raise AssertionError(f'{glacier}: baseline model does not reproduce Code 12. Confirm bulk configuration and inputs.')


def create_figure(all_stats,out_dir):
    plt.rcParams.update({'font.size':FONT_PT,'axes.titlesize':FONT_PT,'axes.labelsize':FONT_PT,
                         'xtick.labelsize':FONT_PT,'ytick.labelsize':FONT_PT,'legend.fontsize':FONT_PT})
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.1),sharex='col')
    keys=[('roughness','Relative_ablation_change_pct'),('height','Relative_ablation_change_pct'),
          ('roughness','Qh_RMSE_W_m2'),('height','Qh_RMSE_W_m2')]
    names=['(a) Roughness length','(b) Measurement height','(c) Roughness length','(d) Measurement height']
    for ax,(experiment,metric),heading in zip(axs.flat,keys,names):
        for glacier in ('Peyto','Athabasca'):
            part=all_stats[(all_stats.glacier==glacier)&(all_stats.experiment==experiment)]
            x=part.value.to_numpy(float)*(1000 if experiment=='roughness' else 1)
            y=part[metric].to_numpy(float)
            ax.plot(x,y,'-',color=COLORS[glacier],
                    linewidth=1.5,label=glacier,zorder=3)
            # Open markers flag provisional upper-level Peyto humidity forcing;
            # applies to both the ablation and Qh-RMSE height panels.
            for xval,yval in zip(x,y):
                provisional=(experiment=='height' and glacier=='Peyto' and xval in (4,6))
                ax.plot([xval],[yval],marker='o',linestyle='None',markersize=5.5,
                        markeredgewidth=1.25,markeredgecolor=COLORS[glacier],
                        markerfacecolor='white' if provisional else COLORS[glacier],zorder=4)
        if experiment=='roughness':ax.set_xscale('log');ax.set_xticks([.5,1,2.41,5,10]);ax.set_xticklabels(['0.5','1','2.41','5','10'])
        if metric.startswith('Relative'):ax.axhline(0,color='0.5',linewidth=0.9)
        ax.set_title(heading,loc='left',fontweight='normal',pad=7)
        ax.grid(True,alpha=0.25)
        ax.tick_params(direction='out')
    for ax in axs[0,:]:ax.set_ylabel('Relative cumulative\nablation change (%)')
    axs[1,0].set_ylabel(r'$Q_h$ RMSE (W m$^{-2}$)')
    axs[1,1].set_ylabel(r'$Q_h$ RMSE (W m$^{-2}$)')
    axs[1,0].set_xlabel(r'Aerodynamic roughness $z_0$ (mm)')
    axs[1,1].set_xlabel('Measurement height (m)')
    axs[0,0].legend(loc='best',frameon=True,edgecolor='0.75')
    fig.subplots_adjust(left=0.12,right=0.985,bottom=0.105,top=0.96,wspace=0.31,hspace=0.30)
    out_dir.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        path=out_dir/f'{STEM}.{ext}'
        fig.savefig(path,dpi=300 if ext=='png' else None,bbox_inches='tight')
        print('Saved',path)
    plt.close(fig)


def main(root_arg=None):
    root=find_root(root_arg)
    interim,atha_raw,figures,tables=source_paths(root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.turbulent_fluxes import bulk_transfer
    p=load_csv(interim/'Peyto_ablation_partition_hourly.csv')
    a=load_csv(interim/'Athabasca_ablation_partition_hourly.csv')
    for glacier,frame in (('Peyto',p),('Athabasca',a)):
        if not frame.index.is_monotonic_increasing:raise AssertionError('Bad time ordering')
    profile_path=atha_raw/'AthaIceProfile.txt'
    if not profile_path.exists():
        paths=list(atha_raw.rglob('AthaIceProfile.txt'))
        if not paths:raise FileNotFoundError(f'Missing Athabasca profile: {profile_path}')
        profile_path=paths[0]
    profile=read_atha_profile(profile_path,a.index)
    profile['U_1p5_filled']=fill_atha_1p5_wind(profile,a)
    all_rows=[]
    for glacier,df in (('Peyto',p),('Athabasca',a)):
        pz=compute_experiment(df,glacier,'roughness',bulk_transfer,profile)
        ph=compute_experiment(df,glacier,'height',bulk_transfer,profile)
        validate_against_code12(df,glacier,pz)
        all_rows.extend([pz,ph])
    result=pd.concat(all_rows,ignore_index=True)
    # Absolute and percentage values are retained for full reproducibility.
    tables.mkdir(parents=True,exist_ok=True)
    stats=tables/f'{STEM}_statistics.csv'
    result.to_csv(stats,index=False)
    print('Saved',stats)
    print(result[['glacier','experiment','value','Ablation_mmwe','Relative_ablation_change_pct',
                  'Qh_RMSE_W_m2','n_Qh_RMSE','n_valid_ablation_hours','n_expected_ablation_hours']].round(4).to_string(index=False))
    create_figure(result,figures)
    return result


if __name__=='__main__':
    if 'ipykernel' in sys.modules:main()
    else:
        parser=argparse.ArgumentParser()
        parser.add_argument('--root',type=str,default=None)
        args=parser.parse_args()
        main(args.root)
