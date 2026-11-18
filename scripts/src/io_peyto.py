"""Readers and preprocessing for the 2008 Peyto profile/EC experiment."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .thermodynamics import saturation_vapor_pressure_hpa

PROFILE_HEIGHTS = (1, 2, 4, 6)


def _doy_hhmm_to_datetime(doy, hhmm, year=2008):
    doy = pd.to_numeric(doy, errors="coerce")
    hhmm = pd.to_numeric(hhmm, errors="coerce")
    base = pd.Timestamp(f"{year}-01-01")
    out = []
    for d, t in zip(doy, hhmm):
        if not np.isfinite(d) or not np.isfinite(t):
            out.append(pd.NaT)
            continue
        s = f"{int(round(t)):04d}"
        hour = int(s[:-2] or 0)
        minute = int(s[-2:])
        out.append(base + pd.Timedelta(days=int(d)-1, hours=hour, minutes=minute))
    return pd.DatetimeIndex(out)


def read_profile_a_tue(path):
    """Read one raw ProfileA_tue_*.xlsx file.

    Column interpretation follows import_profileA.m:
      u: columns 3-6
      T: columns 7-10
      e: columns 11-14
      QH: columns 15-18
      QE: columns 19-22

    The raw vapour-pressure header says hPa, but numerical values are ~Pa.
    The MATLAB code obtained RH as ``e_raw / esat_hPa``; this is numerically
    equivalent to converting e_raw Pa -> hPa and then multiplying the ratio
    by 100 to obtain percent RH. We implement that explicitly.
    """
    path = Path(path)
    raw = pd.read_excel(path, header=None, skiprows=3)
    if raw.shape[1] < 22:
        raise ValueError(f"{path.name}: expected at least 22 columns, got {raw.shape[1]}.")

    dt = _doy_hhmm_to_datetime(raw.iloc[:,0], raw.iloc[:,1])
    df = pd.DataFrame(index=dt)

    for j, h in enumerate(PROFILE_HEIGHTS):
        df[f"u_{h}m"] = pd.to_numeric(raw.iloc[:,2+j], errors="coerce").to_numpy()
        df[f"T_{h}m_C"] = pd.to_numeric(raw.iloc[:,6+j], errors="coerce").to_numpy()
        e_raw_pa = pd.to_numeric(raw.iloc[:,10+j], errors="coerce").to_numpy()
        df[f"e_{h}m_raw_Pa"] = e_raw_pa

        esat_hpa = saturation_vapor_pressure_hpa(df[f"T_{h}m_C"].to_numpy() + 273.15)
        df[f"RH_{h}m_pct"] = e_raw_pa / esat_hpa

        df[f"QH_profile_{h}m"] = pd.to_numeric(raw.iloc[:,14+j], errors="coerce").to_numpy()
        df[f"QE_profile_{h}m"] = pd.to_numeric(raw.iloc[:,18+j], errors="coerce").to_numpy()

    df = df[~df.index.isna()].sort_index()
    df.index.name = "time"
    return df


def concatenate_profile_a(paths):
    frames = [read_profile_a_tue(p) for p in paths]
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def read_crhm_obs(path):
    """Read a CRHM .obs text file with a ##### header line."""
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("################"))
    names = lines[header_idx].split("\t")[1:]
    rows = []
    times = []
    for line in lines[header_idx+1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            times.append(pd.Timestamp(
                year=int(parts[0]), month=int(parts[1]), day=int(parts[2]),
                hour=int(parts[3]), minute=int(parts[4])
            ))
            rows.append([float(x) for x in parts[5:]])
        except Exception:
            continue
    n = min(len(names), len(rows[0]) if rows else 0)
    return pd.DataFrame([r[:n] for r in rows], index=pd.DatetimeIndex(times),
                        columns=names[:n])


def build_profile_a_hourly(profile_paths, moh_obs_path, start="2008-07-24 01:00",
                           end="2008-08-25 00:00"):
    """Recreate the hourly Peyto Profile A forcing used by the MATLAB workflow.

    Reproduces PeytoProfileA_intoCRHMObs.m:
    1. concatenate raw 5-min profile periods;
    2. hourly means;
    3. set RH > 80% at 2/4/6 m to NaN;
    4. fill missing T, RH, U at every level from Peyto MOH;
    5. return the exact manuscript hourly time grid.
    """
    raw = concatenate_profile_a(profile_paths)
    hourly = raw.resample("1h").mean()

    for h in (2, 4, 6):
        hourly.loc[hourly[f"RH_{h}m_pct"] > 80, f"RH_{h}m_pct"] = np.nan

    moh = read_crhm_obs(moh_obs_path)
    moh = moh.resample("1h").mean()
    # Uploaded CRHM files normally expose t.1/rh.1/u.1.
    tcol = next(c for c in moh.columns if c.startswith("t"))
    rhcol = next(c for c in moh.columns if c.startswith("rh"))
    ucol = next(c for c in moh.columns if c.startswith("u"))

    for h in PROFILE_HEIGHTS:
        hourly[f"T_{h}m_C"] = hourly[f"T_{h}m_C"].fillna(moh[tcol])
        hourly[f"RH_{h}m_pct"] = hourly[f"RH_{h}m_pct"].fillna(moh[rhcol])
        hourly[f"u_{h}m"] = hourly[f"u_{h}m"].fillna(moh[ucol])

    idx = pd.date_range(start=start, end=end, freq="1h")
    return hourly.reindex(idx)


def read_ec_qh_qe(path):
    raw = pd.read_excel(path, header=1)
    # Expected: Y M D H M S EC1 Height EC1QE EC2QE EC1QH EC2QH
    t = pd.to_datetime(dict(
        year=pd.to_numeric(raw["Y"], errors="coerce"),
        month=pd.to_numeric(raw["M"], errors="coerce"),
        day=pd.to_numeric(raw["D"], errors="coerce"),
        hour=pd.to_numeric(raw["H"], errors="coerce"),
        minute=pd.to_numeric(raw["M.1"], errors="coerce") if "M.1" in raw.columns else pd.to_numeric(raw.iloc[:,4], errors="coerce"),
        second=pd.to_numeric(raw["S"], errors="coerce"),
    ), errors="coerce")
    out = raw.copy()
    out.index = t
    return out[~out.index.isna()].sort_index()


def read_corrected_qe(path):
    raw = pd.read_excel(path, header=1)
    tcol = raw.columns[0]
    t = pd.to_datetime(raw[tcol], errors="coerce")
    out = raw.copy()
    out.index = t
    return out[~out.index.isna()].sort_index()


def build_ec_hourly(qh_qe_path, corrected_qe_path, start="2008-07-24 01:00",
                    end="2008-08-25 00:00"):
    """Build hourly EC reference matching ComparingIcemelt_Qm_Peyto_v2.m."""
    ec = read_ec_qh_qe(qh_qe_path)
    qec = read_corrected_qe(corrected_qe_path)

    qh = ec[["EC1QH", "EC2QH"]].resample("1h").mean()
    qe = qec[["AdjEC1QE", "AdjEC2QE"]].resample("1h").mean()

    idx = pd.date_range(start=start, end=end, freq="1h")
    out = pd.concat([qh, qe], axis=1).reindex(idx)
    return out
