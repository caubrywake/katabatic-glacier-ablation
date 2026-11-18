"""06 - Bulk-transfer sensitivity to roughness length and measurement height.

This script replaces the ad-hoc sensitivity calculations used earlier in the
manuscript workflow and makes their assumptions explicit.

Outputs
-------
tables/Peyto_z0_sensitivity_stats.csv
tables/Athabasca_z0_sensitivity_stats.csv
tables/Peyto_height_sensitivity_stats.csv
tables/Athabasca_height_sensitivity_stats.csv
figures/Fig09_bulk_sensitivity.png/.pdf

Notes
-----
- Roughness sensitivity uses the PRIMARY bulk forcing at each glacier:
    Peyto: 2 m Toe profile T/RH/U
    Athabasca: ice-AWS T/RH/U (T,RH ~1.5 m; U ~2 m), z=1.5 m
- Height sensitivity uses profile T/RH/U at each available reliable level.
- Peyto EC observations have already been final-QC'd upstream by script 03.
- Athabasca 1.5 m profile wind contains a failed zero/missing period. Those
  values are filled ONLY for the height-sensitivity diagnostic using a linear
  regression against the ice-AWS wind, fitted over overlapping valid non-zero
  profile observations. Valid profile wind observations are never replaced.
- Athabasca 6 m RH is excluded from the primary height-sensitivity analysis
  because the sensor contains physically impossible values (>100%, including
  severe excursions). Athabasca height sensitivity therefore stops at 4.5 m.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_config import PROJECT_ROOT, PEYTO_RAW, ATHA_RAW, INTERIM, TABLES, FIGURES
# Import shared modules from the shipped scripts/ directory (not from a machine-specific root).
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline_common import PROJECT_ROOT, PEYTO_RAW, ATHA_RAW, INTERIM, TABLES, FIGURES, require

sys.path.insert(0, str(PIPELINE_DIR))
from src.turbulent_fluxes import bulk_transfer

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

LF = 3.169e5
Z0_VALUES = [0.0005, 0.001, 0.00241, 0.005, 0.010]
PEYTO_HEIGHTS = [1.0, 2.0, 4.0, 6.0]
ATHA_HEIGHTS_ALL = [0.75, 1.5, 3.0, 4.5, 6.0]
ATHA_HEIGHTS = [0.75, 1.5, 3.0, 4.5]


def metrics(obs, mod, mask_extra=None):
    o = pd.to_numeric(pd.Series(obs), errors="coerce").to_numpy(float)
    m = pd.to_numeric(pd.Series(mod), errors="coerce").to_numpy(float)
    mask = np.isfinite(o) & np.isfinite(m)
    if mask_extra is not None:
        mask &= np.asarray(mask_extra, bool)
    o, m = o[mask], m[mask]
    if len(o) < 2:
        return dict(n=len(o), mean=np.nan, bias=np.nan, rmse=np.nan, r=np.nan)
    return dict(
        n=len(o), mean=float(np.mean(m)), bias=float(np.mean(m-o)),
        rmse=float(np.sqrt(np.mean((m-o)**2))), r=float(np.corrcoef(o,m)[0,1])
    )


def melt_total(qn, qp, qh, qe):
    qn = pd.to_numeric(qn, errors="coerce").to_numpy(float)
    qp = pd.to_numeric(qp, errors="coerce").to_numpy(float)
    qh = np.asarray(qh, float); qe = np.asarray(qe, float)
    qm = qn + qp + qh + qe
    melt = np.where(np.isfinite(qm), np.maximum(qm, 0.0) * 3600.0 / LF, np.nan)
    return float(np.nansum(melt))


def site_common_obs_mask(df, qh_col, qe_col):
    return np.isfinite(pd.to_numeric(df[qh_col], errors="coerce")) & np.isfinite(pd.to_numeric(df[qe_col], errors="coerce"))


def run_z0(df, glacier):
    if glacier == "Peyto":
        T, RH, U, z, alt = df["T_2m_C"], df["RH_2m_pct"], df["u_2m"], 2.0, 2169
        qh_obs, qe_obs = df["EC2QH"], df["AdjEC2QE"]
    else:
        T, RH, U, z, alt = df["T_AWS_C"], df["RH_AWS_pct"], df["U_AWS_ms"], 1.5, 2175
        qh_obs, qe_obs = df["Qh_EC"], df["Qe_EC"]
    obs_mask = site_common_obs_mask(df, qh_obs.name, qe_obs.name)
    rows=[]
    for z0 in Z0_VALUES:
        model=bulk_transfer(pd.to_numeric(T,errors='coerce').to_numpy(float)+273.15,
                            pd.to_numeric(U,errors='coerce').to_numpy(float),
                            pd.to_numeric(RH,errors='coerce').to_numpy(float),
                            altitude_m=alt,z_m=z,z0m=z0)
        mh=metrics(qh_obs,model['Qh'],obs_mask); me=metrics(qe_obs,model['Qe'],obs_mask)
        rows.append({
            'z0_m':z0,'z0_label':f'{z0*1000:g} mm','n':min(mh['n'],me['n']),
            'Qh_mean':mh['mean'],'Qh_bias':mh['bias'],'Qh_RMSE':mh['rmse'],'Qh_r':mh['r'],
            'Qe_mean':me['mean'],'Qe_bias':me['bias'],'Qe_RMSE':me['rmse'],'Qe_r':me['r'],
            'TotalMelt_mmwe':melt_total(df['Qn'],df['Qp'],model['Qh'],model['Qe'])
        })
    out=pd.DataFrame(rows)
    base=float(out.loc[np.isclose(out.z0_m,0.00241),'TotalMelt_mmwe'].iloc[0])
    out['RelativeMeltChange_pct']=(out['TotalMelt_mmwe']/base-1)*100
    return out


def read_atha_profile(target_index):
    path=next(iter(ATHA_RAW.rglob('AthaIceProfile.txt')),None)
    if path is None: raise FileNotFoundError('AthaIceProfile.txt not found')
    d=pd.read_csv(path,sep='\t',skiprows=[0,2,3],na_values=['NAN','NaN','nan'])
    d['time']=pd.to_datetime(d['TIMESTAMP'],errors='coerce')
    out=pd.DataFrame(index=pd.DatetimeIndex(d['time']))
    for i,h in enumerate(ATHA_HEIGHTS_ALL,1):
        tag=str(h).replace('.','p')
        out[f'T_{tag}']=pd.to_numeric(d[f'AirTemp_Avg({i})'],errors='coerce').to_numpy()
        out[f'RH_{tag}']=pd.to_numeric(d[f'RH_Avg({i})'],errors='coerce').to_numpy()
        out[f'U_{tag}']=pd.to_numeric(d[f'WindSpeed_Avg({i})'],errors='coerce').to_numpy()
    out=out[~out.index.isna()].resample('1h').mean().reindex(target_index)
    return out


def run_height_peyto(df):
    obs_mask=site_common_obs_mask(df,'EC2QH','AdjEC2QE'); rows=[]
    for h in PEYTO_HEIGHTS:
        tag=f'{int(h)}m'
        model=bulk_transfer(df[f'T_{tag}_C'].to_numpy(float)+273.15,
                            df[f'u_{tag}'].to_numpy(float),df[f'RH_{tag}_pct'].to_numpy(float),
                            altitude_m=2169,z_m=h)
        mh=metrics(df['EC2QH'],model['Qh'],obs_mask); me=metrics(df['AdjEC2QE'],model['Qe'],obs_mask)
        rows.append({'height_m':h,'n':min(mh['n'],me['n']),'Qh_r':mh['r'],'Qe_r':me['r'],
                     'Qh_mean':mh['mean'],'Qh_bias':mh['bias'],'Qh_RMSE':mh['rmse'],
                     'Qe_mean':me['mean'],'Qe_bias':me['bias'],'Qe_RMSE':me['rmse'],
                     'TotalMelt_mmwe':melt_total(df['Qn'],df['Qp'],model['Qh'],model['Qe'])})
    out=pd.DataFrame(rows)
    base=float(out.loc[out.height_m==2.0,'TotalMelt_mmwe'].iloc[0]); out['RelativeMeltChange_pct']=(out.TotalMelt_mmwe/base-1)*100
    return out


def linear_regression_fill_source(x, y):
    """Fit y = intercept + slope*x and return coefficients and diagnostics."""
    x = pd.to_numeric(x, errors="coerce").to_numpy(float)
    y = pd.to_numeric(y, errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        raise ValueError("Not enough paired observations for regression.")
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    r = float(np.corrcoef(x, y)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    bias_raw = float(np.mean(y - x))
    rmse_raw = float(np.sqrt(np.mean((y - x) ** 2)))
    return {
        "n": int(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": r,
        "r2": r * r,
        "regression_rmse_m_s": rmse,
        "raw_bias_profile_minus_AWS_m_s": bias_raw,
        "raw_rmse_profile_minus_AWS_m_s": rmse_raw,
    }


def run_height_atha(df):
    prof = read_atha_profile(df.index)
    obs_mask = site_common_obs_mask(df, 'Qh_EC', 'Qe_EC')

    # ---------------------------------------------------------
    # QC 1: 1.5 m wind sensor
    # ---------------------------------------------------------
    # The profile 1.5 m anemometer contains exact zeros / missing data.
    # Calibrate the AWS (~2 m) wind to the valid 1.5 m profile wind and
    # use the corrected AWS estimate ONLY to fill failed profile values.
    u_profile = pd.to_numeric(prof['U_1p5'], errors='coerce')
    u_aws = pd.to_numeric(df['U_AWS_ms'], errors='coerce')

    valid_reg = (
        np.isfinite(u_profile)
        & (u_profile > 0)
        & np.isfinite(u_aws)
    )
    reg = linear_regression_fill_source(u_aws[valid_reg], u_profile[valid_reg])

    bad_1p5 = (~np.isfinite(u_profile)) | (u_profile <= 0)
    u_1p5_filled = u_profile.copy()
    u_1p5_filled.loc[bad_1p5 & np.isfinite(u_aws)] = (
        reg["intercept"] + reg["slope"] * u_aws.loc[bad_1p5 & np.isfinite(u_aws)]
    )
    # Wind cannot be negative after regression correction.
    u_1p5_filled = u_1p5_filled.clip(lower=0.0)
    prof['U_1p5_filled'] = u_1p5_filled

    reg_row = {
        "comparison": "profile_1p5m_vs_AWS_wind",
        **reg,
        "n_bad_profile_hours": int(bad_1p5.sum()),
        "n_filled_from_AWS": int((bad_1p5 & np.isfinite(u_aws)).sum()),
    }

    print("\nAthabasca 1.5 m wind QC / AWS regression:")
    for k, v in reg_row.items():
        print(f"  {k}: {v}")

    # ---------------------------------------------------------
    # QC 2: 6 m RH sensor
    # ---------------------------------------------------------
    # Report the problem explicitly, but do not use 6 m in the primary
    # Athabasca height-sensitivity analysis.
    rh45 = pd.to_numeric(prof['RH_4p5'], errors='coerce')
    rh6 = pd.to_numeric(prof['RH_6p0'], errors='coerce')
    good_rh6 = np.isfinite(rh45) & np.isfinite(rh6) & (rh6 >= 0) & (rh6 <= 100)

    if good_rh6.sum() >= 3:
        rh_reg = linear_regression_fill_source(rh45[good_rh6], rh6[good_rh6])
    else:
        rh_reg = {
            "n": int(good_rh6.sum()), "slope": np.nan, "intercept": np.nan,
            "r": np.nan, "r2": np.nan, "regression_rmse_m_s": np.nan,
            "raw_bias_profile_minus_AWS_m_s": np.nan,
            "raw_rmse_profile_minus_AWS_m_s": np.nan,
        }

    bad_rh6 = np.isfinite(rh6) & ((rh6 < 0) | (rh6 > 100))
    print("\nAthabasca 6 m RH QC:")
    print("  physically invalid hourly RH values:", int(bad_rh6.sum()))
    print("  valid 4.5/6 m RH pairs:", int(good_rh6.sum()))
    print("  valid-pair r:", rh_reg["r"])
    print("  valid-pair R2:", rh_reg["r2"])
    print("  6 m RH is excluded from primary Figure 9 height sensitivity.")

    qc = pd.DataFrame([
        reg_row,
        {
            "comparison": "RH_6m_vs_RH_4p5m_valid_subset",
            "n": rh_reg["n"],
            "slope": rh_reg["slope"],
            "intercept": rh_reg["intercept"],
            "r": rh_reg["r"],
            "r2": rh_reg["r2"],
            "regression_rmse_m_s": np.nan,
            "raw_bias_profile_minus_AWS_m_s": np.nan,
            "raw_rmse_profile_minus_AWS_m_s": np.nan,
            "n_bad_profile_hours": int(bad_rh6.sum()),
            "n_filled_from_AWS": 0,
        }
    ])
    qc.to_csv(TABLES / "Athabasca_profile_height_QC.csv", index=False)

    # ---------------------------------------------------------
    # Primary height sensitivity: 0.75, 1.5, 3, 4.5 m only
    # ---------------------------------------------------------
    rows = []
    for h in ATHA_HEIGHTS:
        tag = str(h).replace('.', 'p')
        T = prof[f'T_{tag}']
        RH = prof[f'RH_{tag}']
        if h == 1.5:
            U = prof['U_1p5_filled']
        else:
            U = prof[f'U_{tag}']

        model = bulk_transfer(
            T.to_numpy(float) + 273.15,
            U.to_numpy(float),
            RH.to_numpy(float),
            altitude_m=2175,
            z_m=h
        )
        mh = metrics(df['Qh_EC'], model['Qh'], obs_mask)
        me = metrics(df['Qe_EC'], model['Qe'], obs_mask)
        rows.append({
            'height_m': h,
            'n': min(mh['n'], me['n']),
            'Qh_r': mh['r'], 'Qe_r': me['r'],
            'Qh_mean': mh['mean'], 'Qh_bias': mh['bias'], 'Qh_RMSE': mh['rmse'],
            'Qe_mean': me['mean'], 'Qe_bias': me['bias'], 'Qe_RMSE': me['rmse'],
            'TotalMelt_mmwe': melt_total(df['Qn'], df['Qp'], model['Qh'], model['Qe'])
        })

    out = pd.DataFrame(rows)
    base = float(out.loc[out.height_m == 1.5, 'TotalMelt_mmwe'].iloc[0])
    out['RelativeMeltChange_pct'] = (out.TotalMelt_mmwe / base - 1) * 100
    return out


def make_figure(pz,az,ph,ah):
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    axs[0,0].semilogx(pz.z0_m*1000,pz.RelativeMeltChange_pct,'o-',label='Peyto'); axs[0,0].semilogx(az.z0_m*1000,az.RelativeMeltChange_pct,'o-',label='Athabasca')
    axs[0,0].axhline(0,color='0.5',lw=.8); axs[0,0].set_xlabel(r'$z_0$ (mm)'); axs[0,0].set_ylabel('Relative melt change (%)'); axs[0,0].set_title('(a) Roughness length: melt'); axs[0,0].legend()
    axs[1,0].semilogx(pz.z0_m*1000,pz.Qh_RMSE,'o-',label='Peyto'); axs[1,0].semilogx(az.z0_m*1000,az.Qh_RMSE,'o-',label='Athabasca'); axs[1,0].set_xlabel(r'$z_0$ (mm)'); axs[1,0].set_ylabel(r'$Q_h$ RMSE (W m$^{-2}$)'); axs[1,0].set_title('(c) Roughness length: Qh RMSE')
    axs[0,1].plot(ph.height_m,ph.RelativeMeltChange_pct,'o-',label='Peyto'); axs[0,1].plot(ah.height_m,ah.RelativeMeltChange_pct,'o-',label='Athabasca'); axs[0,1].axhline(0,color='0.5',lw=.8); axs[0,1].set_xlabel('Measurement height (m)'); axs[0,1].set_ylabel('Relative melt change (%)'); axs[0,1].set_title('(b) Measurement height: melt')
    axs[1,1].plot(ph.height_m,ph.Qh_RMSE,'o-',label='Peyto'); axs[1,1].plot(ah.height_m,ah.Qh_RMSE,'o-',label='Athabasca'); axs[1,1].set_xlabel('Measurement height (m)'); axs[1,1].set_ylabel(r'$Q_h$ RMSE (W m$^{-2}$)'); axs[1,1].set_title('(d) Measurement height: Qh RMSE')
    for ax in axs.flat: ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURES/'Fig09_bulk_sensitivity.png',dpi=300,bbox_inches='tight'); fig.savefig(FIGURES/'Fig09_bulk_sensitivity.pdf',bbox_inches='tight'); plt.close(fig)


def main():
    p=pd.read_csv(require(INTERIM/'Peyto_melt_scenarios_hourly.csv'),parse_dates=['time']).set_index('time')
    a=pd.read_csv(require(INTERIM/'Athabasca_melt_scenarios_hourly.csv'),parse_dates=['time']).set_index('time')
    pz,az=run_z0(p,'Peyto'),run_z0(a,'Athabasca')
    ph,ah=run_height_peyto(p),run_height_atha(a)
    for name,d in [('Peyto_z0_sensitivity_stats.csv',pz),('Athabasca_z0_sensitivity_stats.csv',az),('Peyto_height_sensitivity_stats.csv',ph),('Athabasca_height_sensitivity_stats.csv',ah)]:
        d.to_csv(TABLES/name,index=False); print('\n',name,'\n',d.to_string(index=False))
    make_figure(pz,az,ph,ah)
    print('\nSaved Figure 9, four sensitivity tables, and Athabasca_profile_height_QC.csv.')

if __name__=='__main__': main()
