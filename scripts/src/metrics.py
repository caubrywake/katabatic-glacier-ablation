from __future__ import annotations
import numpy as np
import pandas as pd


def pairwise_metrics(obs, mod):
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() == 0:
        return {"n": 0, "mean_obs": np.nan, "mean_mod": np.nan,
                "bias": np.nan, "rmse": np.nan, "r": np.nan,
                "sd_obs": np.nan, "sd_mod": np.nan}
    o, y = obs[m], mod[m]
    r = np.corrcoef(o, y)[0,1] if len(o) > 1 else np.nan
    return {
        "n": int(m.sum()),
        "mean_obs": float(np.mean(o)),
        "mean_mod": float(np.mean(y)),
        "bias": float(np.mean(y-o)),
        "rmse": float(np.sqrt(np.mean((y-o)**2))),
        "r": float(r),
        "sd_obs": float(np.std(o, ddof=1)) if len(o)>1 else np.nan,
        "sd_mod": float(np.std(y, ddof=1)) if len(y)>1 else np.nan,
    }


def flux_summary(df, ec_qh="EC2QH", ec_qe="AdjEC2QE"):
    rows = []
    for flux, obs_col, mods in [
        ("Qh", ec_qh, ["Qh_BT","Qh_Kata"]),
        ("Qe", ec_qe, ["Qe_BT","Qe_Kata"]),
    ]:
        for mod_col in mods:
            x = pairwise_metrics(df[obs_col], df[mod_col])
            x.update({"flux": flux, "model": mod_col.replace(f"{flux}_","")})
            rows.append(x)
    return pd.DataFrame(rows)
